from torch.nn import functional as F
import torch
from torch import nn
from torch_geometric.nn import GCNConv, ChebConv  # noqa
from .relation import Relationshipness


class RelationEncoder(nn.Module):
    def __init__(self, padding_idx=0, d_in=2048, d_model=512, box_dim=8, dropout=0.1, rela_threshold=0.5, use_relaAtt=True, use_pos=True):
        super(RelationEncoder, self).__init__()
        self.d_model = d_model
        self.use_relaAtt = use_relaAtt
        self.use_pos = use_pos
        self.dropout = dropout
        self.padding_idx = padding_idx
        self.rela_threshold = rela_threshold
        # pre-process
        self.preprocess = nn.Sequential(
            nn.Linear(d_in, self.d_model),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.LayerNorm(self.d_model)
        )
        self.subj_proj = nn.Sequential(
            nn.Linear(self.d_model, self.d_model) if self.use_relaAtt else nn.Linear(
                self.d_model, 64),
            nn.ReLU(),
            nn.Dropout(p=self.dropout)
        )

        self.obj_proj = nn.Sequential(
            nn.Linear(self.d_model, self.d_model) if self.use_relaAtt else nn.Linear(
                self.d_model, 64),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
        )

        if self.use_pos:
            self.pos_self = nn.Sequential(
                nn.Linear(box_dim, 64),
                nn.ReLU(True),
                nn.Dropout(p=self.dropout),
                nn.LayerNorm(64),
                nn.Linear(64, 64),
                nn.ReLU(True),
                nn.Dropout(p=self.dropout),
            )
            self.pos_relas = nn.Sequential(
                nn.Linear(box_dim, 64),
                nn.ReLU(True),
                nn.Dropout(p=self.dropout),
                nn.LayerNorm(64),
                nn.Linear(64, 64),
                nn.ReLU(True),
                nn.Dropout(p=self.dropout),
            )
        # gcn
        self.relation = Relationshipness(
            use_relaAtt=self.use_relaAtt, use_pos=self.use_pos)

        self.gcn1 = GCNConv(self.d_model+80, self.d_model+80)
        self.relu1 = nn.ReLU(True)
        self.dropout1 = nn.Dropout(p=self.dropout*3)
        self.gcn2 = GCNConv(self.d_model+80, self.d_model)
        self.relu2 = nn.ReLU(True)
        self.dropout2 = nn.Dropout(p=self.dropout*3)
        self.gcn3 = GCNConv(self.d_model, self.d_model)
        self.relu3 = nn.ReLU(True)
        self.dropout3 = nn.Dropout(p=self.dropout*3)

    def forward(self, images, selfbbox, bbox):
        # 10,50,2048->10,50,512
        nums = images.shape[1]
        index = list(range(nums))
        mask = (torch.sum(images, -1) != self.padding_idx)
        attention_mask = (torch.sum(images, -1) == self.padding_idx).unsqueeze(1).unsqueeze(1)
        x = self.preprocess(images)
        ##########################################################
        # 关系网络

        # 为 从特征找关系 准备数据
        # 两个核函数映射 10,50,512->10,50,512
        x_subj = self.subj_proj(x)
        x_obj = self.obj_proj(x)

        x_subj = x_subj.masked_fill(mask.unsqueeze(-1) == 0, 0.)
        x_obj = x_obj.masked_fill(mask.unsqueeze(-1) == 0, 0.)
        # 16，50，50，512
        x_rela = x_obj.unsqueeze(1).repeat(1, nums, 1, 1)
        x_rela[:, index, index, :] = x_subj[:, index, :]
        x_rela = x_rela.masked_fill(mask.unsqueeze(-1).unsqueeze(-1) == 0, 0.)
        # 为 从位置找关系 准备数据
        pos_self = self.pos_self(selfbbox)
        pos_relas = self.pos_relas(bbox)

        pos_self = pos_self.masked_fill(mask.unsqueeze(-1) == 0, 0.)
        pos_relas = pos_relas.masked_fill(mask.unsqueeze(
            1).unsqueeze(-1) == 0, 0.).masked_fill(mask.unsqueeze(-1).unsqueeze(-1) == 0, 0.)
        # 16，50，50，8
        pos_rela = pos_relas
        pos_rela[:, index, index, :] = pos_self[:, index, :]
        pos_rela = pos_rela.masked_fill(
            mask.unsqueeze(-1).unsqueeze(-1) == 0, 0.)
        # 为每一个 特征 学习一个与其他所有特征有关的 关系特征 和其 概率
        # input: 512+64   ouput: 16,50,80(64+16)
        if self.use_pos:
            relas, p = self.relation(
                torch.cat((x_rela, pos_rela), dim=-1), mask)
        else:
            relas, p = self.relation(x_rela, mask)
        ##########################################################

        xx = torch.cat((x, relas), dim=-1)
        ##########################################################
        # gcn
        chunk, nums = xx.shape[0], xx.shape[1]
        index = torch.nonzero(p > self.rela_threshold)
        edge_index = torch.stack(
            (index[:, 1], index[:, 2]), dim=0)+index[:, 0]*nums
        chunk_relas = [one.squeeze(0)
                       for one in torch.chunk(xx, chunk, dim=0)]
        gcn_x = torch.cat(chunk_relas, dim=0)
        x1 = self.dropout1(self.relu1(self.gcn1(gcn_x, edge_index)))
        x2 = self.dropout2(self.relu2(self.gcn2(x1, edge_index)))
        xx = self.dropout3(self.relu3(self.gcn3(x2, edge_index)))

        outs = []
        outs.append(x.unsqueeze(1))
        outs.append(torch.stack(xx.split(nums, dim=0), dim=0).unsqueeze(1))
        outs = torch.cat(outs, 1)
        return outs, attention_mask


if __name__ == "__main__":
    imgs = torch.randn(16, 50, 2048)
    selfbox = torch.randn(16, 50, 8)
    box = torch.randn(16, 50, 50, 8)
    rela = RelationEncoder()
    mask = torch.ones(16, 50)
    rela(imgs, selfbox, box)
