import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from models.transformer.attention import MultiHeadAttention
from .basedecoder import BaseDecoder

class TriLSTM(BaseDecoder):
    def __init__(self, vocab_size, padding_idx, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.3,
                 self_att_module=None, enc_att_module=None, self_att_module_kwargs=None, enc_att_module_kwargs=None):
        super(TriLSTM, self).__init__()
        self.d_model = d_model
        self.padding_idx = padding_idx
        
        self.word_emb = nn.Embedding(
            vocab_size, d_model, padding_idx=padding_idx)
        self.vocab_size = vocab_size
        # 输出
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(d_model, vocab_size)

        # 上
        self.att_lstm = nn.LSTMCell(
            self.d_model*3, self.d_model)  # we, fc, h^2_t-1
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

        

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(3, bsz, 512),
                weight.new_zeros(3, bsz, 512))

    def _forward(self, seq, encoder_output, mask_encoder):
        # input (b_s, seq_len)
        seq = self.word_emb(input)
        b_s, seq_len = seq.shape[:2]

        state = self.init_hidden(b_s)

        outputs = encoder_output.new_zeros(
            b_s, seq_len - 1, self.vocab_size)

        for i in range(seq_len - 1):

            it = seq[:, i].clone()
            # break if all the sequences end
            if i >= 1 and seq[:, i].sum() == 0:
                break
            ####################################################
            prev_h1, prev_h2 = state[0][1], state[0][2]
            base_feats, gcn_feats = torch.split(encoder_output, 1, dim=1)
            base_feats, gcn_feats = base_feats.squeeze(1), gcn_feats.squeeze(1)
            fc_feats = torch.sum(base_feats, dim=1)/base_feats.shape[1]

            att_lstm_input = torch.cat([it, fc_feats, prev_h1, prev_h2], 1)

            h_lang, c_lang = self.lang_lstm(
                att_lstm_input, (state[0][0], state[1][0]))

            # h_att； 融合过的节点特征，融合过的边特征
            gcn_att = self.attention_middle(
                gcn_feats, gcn_feats, gcn_feats, attention_mask=mask_encoder)
            gcn_att = torch.sum(gcn_att, dim=1)

            gcn_lstm_input = torch.cat([gcn_att, h_lang], 1)

            h_gcn, c_gcn = self.gcn_lstm(
                gcn_lstm_input, (state[0][1], state[1][1]))

            base_att = self.attention_top(
                base_feats, base_feats, base_feats, attention_mask=mask_encoder)

            base_att = torch.sum(base_att, dim=1)

            ctx_input = torch.cat([base_att, h_gcn, h_lang], 1)

            h_att, c_att = self.att_lstm(
                ctx_input, (state[0][2], state[1][2]))

            state = (torch.stack([h_lang, h_gcn, h_att]),
                     torch.stack([c_lang, c_gcn, c_att]))

            logprobs = F.log_softmax(self.fc(self.dropout(h_att)), dim=-1)

            outputs[:, i] = logprobs

        return outputs

seq = torch.randint(0,30,(16,13))
encoder_output = torch.rand(16,2,50,512)
mask_encoder = torch.ones(16,50)

model = TriLSTM()
model(seq,encoder_output,mask_encoder)
