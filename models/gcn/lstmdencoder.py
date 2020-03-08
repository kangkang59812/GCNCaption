import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from models.transformer.attention import MultiHeadAttention
from models.transformer.utils import sinusoid_encoding_table, PositionWiseFeedForward
from models.containers import Module, ModuleList
from .basedecoder import BaseDecoder


class MeshedDecoderLayer(Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, self_att_module=None,
                 enc_att_module=None, self_att_module_kwargs=None, enc_att_module_kwargs=None):
        super(MeshedDecoderLayer, self).__init__()

        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout)

        self.fc_alpha1 = nn.Linear(d_model + d_model, d_model)
        self.fc_alpha2 = nn.Linear(d_model + d_model, d_model)
        #self.fc_alpha3 = nn.Linear(d_model + d_model, d_model)

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc_alpha1.weight)
        nn.init.xavier_uniform_(self.fc_alpha2.weight)
        # nn.init.xavier_uniform_(self.fc_alpha3.weight)
        nn.init.constant_(self.fc_alpha1.bias, 0)
        nn.init.constant_(self.fc_alpha2.bias, 0)
        #nn.init.constant_(self.fc_alpha3.bias, 0)

    def forward(self, input, enc_output, mask_pad, mask_self_att, mask_enc_att):
        self_att = self.self_att(input, input, input, mask_self_att)
        self_att = self_att * mask_pad

        enc_att1 = self.enc_att(
            self_att, enc_output[:, 0], enc_output[:, 0], mask_enc_att) * mask_pad
        enc_att2 = self.enc_att(
            self_att, enc_output[:, 1], enc_output[:, 1], mask_enc_att) * mask_pad
        # enc_att3 = self.enc_att(
        #     self_att, enc_output[:, 2], enc_output[:, 2], mask_enc_att) * mask_pad

        alpha1 = torch.sigmoid(self.fc_alpha1(
            torch.cat([self_att, enc_att1], -1)))
        alpha2 = torch.sigmoid(self.fc_alpha2(
            torch.cat([self_att, enc_att2], -1)))
        # alpha3 = torch.sigmoid(self.fc_alpha3(
        #     torch.cat([self_att, enc_att3], -1)))

        enc_att = (enc_att1 * alpha1 + enc_att2 *
                   alpha2) / np.sqrt(2)
        enc_att = enc_att * mask_pad

        ff = self.pwff(enc_att)
        ff = ff * mask_pad
        return ff


class TriLSTM(BaseDecoder):
    def __init__(self, vocab_size, max_len, padding_idx, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1,
                 self_att_module=None, enc_att_module=None, self_att_module_kwargs=None, enc_att_module_kwargs=None):
        super(TriLSTM, self).__init__()
        self.d_model = d_model
        self.word_emb = nn.Embedding(
            vocab_size, d_model, padding_idx=padding_idx)

        # 输出
        self.fc = nn.Linear(d_model, vocab_size, bias=False)

        # 上
        self.att_lstm = nn.LSTMCell(
            self.d_model*3, self.d_model*2)  # we, fc, h^2_t-1
        self.attention_top = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=False,
                                                attention_module=enc_att_module,               attention_module_kwargs=enc_att_module_kwargs)
        # 中
        self.gcn_lstm = nn.LSTMCell(
            self.d_model*2, self.d_model)  # gcn_v, h1t
        self.attention_middle = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=False,
                                                   attention_module=enc_att_module,               attention_module_kwargs=enc_att_module_kwargs)
        # 下
        self.lang_lstm = nn.LSTMCell(
            self.d_model*4, self.d_model)  # h2t-1,h3t-1,x,v平均
        self.self_att = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=True,
                                           attention_module=self_att_module,
                                           attention_module_kwargs=self_att_module_kwargs)

        self.max_len = max_len
        self.padding_idx = padding_idx

    def _forward(self, encoder_output, mask_encoder):
        # input (b_s, seq_len)
        b_s, seq_len = input.shape[:2]
        # (b_s, seq_len, 1)
        mask_queries = (input != self.padding_idx).unsqueeze(-1).float()
        mask_self_attention = torch.triu(torch.ones(
            (seq_len, seq_len), dtype=torch.uint8, device=input.device), diagonal=1)
        mask_self_attention = mask_self_attention.unsqueeze(
            0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
        mask_self_attention = mask_self_attention + \
            (input == self.padding_idx).unsqueeze(1).unsqueeze(1).byte()
        mask_self_attention = mask_self_attention.gt(
            0)  # (b_s, 1, seq_len, seq_len)
        if self._is_stateful:
            self.running_mask_self_attention = torch.cat(
                [self.running_mask_self_attention, mask_self_attention], -1)
            mask_self_attention = self.running_mask_self_attention

        seq = torch.arange(1, seq_len + 1).view(1, -
                                                1).expand(b_s, -1).to(input.device)  # (b_s, seq_len)
        seq = seq.masked_fill(mask_queries.squeeze(-1) == 0, 0)
        if self._is_stateful:
            self.running_seq.add_(1)
            seq = self.running_seq

        out = self.word_emb(input)
        self_att = self.self_att(input, input, input, mask_self_attention)
        self_att = self_att * mask_queries

        out = l(out, encoder_output,
                mask_queries, mask_self_attention, mask_encoder)

        out = self.fc(out)
        return F.log_softmax(out, dim=-1)
