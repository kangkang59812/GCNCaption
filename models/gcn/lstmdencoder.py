import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from models.transformer.attention import MultiHeadAttention
from .basedecoder import BaseDecoder
from .gcnutils import repeat_tensors
from models.containers import Module, ModuleList


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.rnn_size = 512
        self.att_hid_size = 512

        self.project = nn.Linear(self.att_hid_size, self.att_hid_size)
        self.h2att = nn.Linear(self.rnn_size, self.att_hid_size)
        self.alpha_net = nn.Linear(self.att_hid_size, 1)

    def forward(self, h, att_feats, p_att_feats, att_masks=None):
        # The p_att_feats here is already projected
        p_att_feats = self.project(p_att_feats)
        att_size = att_feats.numel() // att_feats.size(0) // att_feats.size(-1)
        att = p_att_feats.view(-1, att_size, self.att_hid_size)

        att_h = self.h2att(h)                        # batch * att_hid_size
        # batch * att_size * att_hid_size
        att_h = att_h.unsqueeze(1).expand_as(att)
        # batch * att_size * att_hid_size
        dot = att + att_h
        # batch * att_size * att_hid_size
        dot = torch.tanh(dot)
        # (batch * att_size) * att_hid_size
        dot = dot.view(-1, self.att_hid_size)
        # (batch * att_size) * 1
        dot = self.alpha_net(dot)
        dot = dot.view(-1, att_size)                        # batch * att_size

        # batch * att_size
        weight = F.softmax(dot, dim=1)
        if att_masks is not None:
            weight = weight * att_masks.view(-1, att_size).float()
            weight = weight / weight.sum(1, keepdim=True)  # normalize to 1
        # batch * att_size * att_feat_size
        att_feats_ = att_feats.view(-1, att_size, att_feats.size(-1))
        att_res = torch.bmm(weight.unsqueeze(1), att_feats_).squeeze(
            1)  # batch * att_feat_size

        return att_res


class TriLSTM(Module):
    def __init__(self, vocab_size, padding_idx, max_len=20, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.3,
                 self_att_module=None, enc_att_module=None, self_att_module_kwargs=None, enc_att_module_kwargs=None):
        super(TriLSTM, self).__init__()
        self.d_model = d_model
        self.padding_idx = padding_idx
        self.seq_length = max_len
        self.word_emb = nn.Embedding(
            vocab_size, d_model, padding_idx=padding_idx)
        self.vocab_size = vocab_size
        # 输出
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(d_model, vocab_size)

        # 上
        self.att_lstm = nn.LSTMCell(
            self.d_model*3, self.d_model)  # we, fc, h^2_t-1
        self.attention_top = Attention()
        # 中
        self.gcn_lstm = nn.LSTMCell(
            self.d_model*2, self.d_model)  # gcn_v, h1t
        self.attention_middle = Attention()
        # 下
        self.lang_lstm = nn.LSTMCell(
            self.d_model*4, self.d_model)  # h2t-1,h3t-1,x,v平均

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(3, bsz, 512),
                weight.new_zeros(3, bsz, 512))

    def forward(self, seq, encoder_output, mask_encoder, state=None, mode='forward'):
        # input (b_s, seq_len)
        if mode == 'forward':
            b_s, seq_len = seq.shape[:2]
            eos = seq.new_ones(b_s)*3
            state = self.init_hidden(b_s)

            outputs = encoder_output.new_zeros(
                b_s, seq_len, self.vocab_size)

            for i in range(seq_len):

                it = seq[:, i].clone()
                # break if all the sequences end, eos=3
                if i >= 1 and (it.float()-eos.float()).sum() == 0:
                    break
                ####################################################
                it = self.word_emb(it)
                output, state = self.get_logprobs_state(
                    it, encoder_output, mask_encoder, state)

                outputs[:, i] = output

            return outputs

        elif mode == 'feedback':
            if state is None:
                b_s, seq_len = seq.shape[:2]
                state = self.init_hidden(b_s)
            it = self.word_emb(seq).squeeze(1)
            logprobs, state = self.get_logprobs_state(
                it, encoder_output, mask_encoder, state)
            return logprobs, state

    def get_logprobs_state(self, seq, encoder_output, mask_encoder, state):

        nums = encoder_output.shape[2]
        prev_h1, prev_h2 = state[0][1], state[0][2]
        base_feats, gcn_feats = torch.split(encoder_output, 1, dim=1)
        base_feats, gcn_feats = base_feats.squeeze(1), gcn_feats.squeeze(1)
        fc_feats = torch.sum(base_feats, dim=1)/base_feats.shape[1]

        att_lstm_input = torch.cat([seq, fc_feats, prev_h1, prev_h2], 1)

        h_lang, c_lang = self.lang_lstm(
            att_lstm_input, (state[0][0], state[1][0]))

        # 顺序问题
        base_att = self.attention_top(
            h_lang, base_feats, base_feats, mask_encoder)

        #base_att = torch.sum(base_att, dim=1)/nums

        ctx_input = torch.cat([base_att, h_lang, h_lang], 1)

        h_att, c_att = self.att_lstm(ctx_input, (state[0][2], state[1][2]))

        gcn_att = self.attention_middle(
            h_att, gcn_feats, gcn_feats, mask_encoder)
        #gcn_att = torch.sum(gcn_att, dim=1)/nums

        gcn_lstm_input = torch.cat([gcn_att, h_att], 1)

        h_gcn, c_gcn = self.gcn_lstm(
            gcn_lstm_input, (state[0][1], state[1][1]))

        

        # 顺序问题
        state = (torch.stack([h_lang, h_gcn, h_att]),
                 torch.stack([c_lang, c_gcn, c_att]))

        logprobs = F.log_softmax(self.fc(self.dropout(h_att)), dim=-1)

        return logprobs, state
