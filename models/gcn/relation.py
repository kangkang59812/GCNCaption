import torch
import torch.nn as nn
from .relaAttention import RelationMultiHeadAttention


class Relationshipness(nn.Module):
    """
    compute relationshipness between subjects and objects
    """

    def __init__(self, use_relaAtt=True, use_pos=True):
        super(Relationshipness, self).__init__()
        self.use_attention = use_relaAtt
        self.pos_encoding = use_pos

        if self.use_attention:
            if self.pos_encoding:
                self.rela = RelationMultiHeadAttention(
                    n_head=8, d_model=512+64, d_k=72, d_v=72)
                self.fc = nn.Linear(512+64, 80)
            else:
                self.rela = RelationMultiHeadAttention(
                    n_head=8, d_model=512, d_k=64, d_v=64)
                self.fc = nn.Linear(512, 80)
            self.norm = nn.LayerNorm(80)
        else:
            if self.pos_encoding:
                self.rela = nn.Linear(64+64, 1)
                self.fc = nn.Linear(128, 80)
            else:
                self.rela = nn.Linear(64, 1)
                self.fc = nn.Linear(64, 80)

        self.dropout = nn.Dropout(0.1)

        # self.alpha =
    def forward(self, x, mask):

        if self.use_attention:
            # 包含了有无pos
            x_rela, rela_p = self.rela(x, x, x, mask)
        else:
            if mask is not None:
                rela_p = self.rela(x_rela).squeeze(-1).masked_fill(
                    mask.unsqueeze(1) == 0, 0.).masked_fill(mask.unsqueeze(-1) == 0, 0.)
            else:
                rela_p = self.rela(x_rela).squeeze(-1)
            x_rela = x

        if mask is not None:
            p = torch.sigmoid(rela_p).masked_fill(mask.unsqueeze(1)
                                                  == 0, 0.).masked_fill(mask.unsqueeze(-1) == 0, 0.)
            x_rela = self.dropout(self.fc(
                (p.unsqueeze(-1)*x_rela).sum(dim=2)).masked_fill(mask.unsqueeze(-1) == 0, 0.))

        else:
            p = torch.sigmoid(rela_p)
            x_rela = self.dropout(self.fc(
                (p.unsqueeze(-1)*x_rela).sum(dim=2)))

        x_rela = self.norm(x_rela)

        return x_rela, p


if __name__ == "__main__":
    att = torch.randn(16, 50, 2048)
    bbox = torch.randn(16, 50, 50, 576)
    m = Relationshipness(use_pos=True, use_relaAtt=True)
    m(bbox, mask=None)
    print('')
