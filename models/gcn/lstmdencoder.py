import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from models.transformer.attention import MultiHeadAttention
from .basedecoder import BaseDecoder
from .gcnutils import repeat_tensors


class Attention(nn.Module):
    def __init__(self, opt):
        super(Attention, self).__init__()
        self.rnn_size = opt.rnn_size
        self.att_hid_size = opt.att_hid_size

        self.h2att = nn.Linear(self.rnn_size, self.att_hid_size)
        self.alpha_net = nn.Linear(self.att_hid_size, 1)

    def forward(self, h, att_feats, p_att_feats, att_masks=None):
        # The p_att_feats here is already projected
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


class TriLSTM(BaseDecoder):
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

        b_s, seq_len = seq.shape[:2]
        eos = seq.new_ones(b_s)*3
        state = self.init_hidden(b_s)

        outputs = encoder_output.new_zeros(
            b_s, seq_len, self.vocab_size)

        for i in range(seq_len):

            it = seq[:, i].clone()
            # break if all the sequences end, eos=3
            if i >= 1 and (seq[:, i]-eos).sum() == 0:
                break
            ####################################################

            output, state = self.get_logprobs_state(
                it, encoder_output, mask_encoder, state)

            outputs[:, i] = output

        return outputs

    def get_logprobs_state(self, it, encoder_output, mask_encoder, state):
        seq = self.word_emb(it)
        nums = encoder_output.shape[2]
        prev_h1, prev_h2 = state[0][1], state[0][2]
        base_feats, gcn_feats = torch.split(encoder_output, 1, dim=1)
        base_feats, gcn_feats = base_feats.squeeze(1), gcn_feats.squeeze(1)
        fc_feats = torch.sum(base_feats, dim=1)/base_feats.shape[1]

        att_lstm_input = torch.cat([seq, fc_feats, prev_h1, prev_h2], 1)

        h_lang, c_lang = self.lang_lstm(
            att_lstm_input, (state[0][0], state[1][0]))

        gcn_att = self.attention_middle(
            h_lang.unsqueeze(1).repeat(1, nums, 1), gcn_feats, gcn_feats, attention_mask=mask_encoder)
        gcn_att = torch.sum(gcn_att, dim=1)/nums

        gcn_lstm_input = torch.cat([gcn_att, h_lang], 1)

        h_gcn, c_gcn = self.gcn_lstm(
            gcn_lstm_input, (state[0][1], state[1][1]))

        base_att = self.attention_top(
            h_gcn.unsqueeze(1).repeat(1, nums, 1), base_feats, base_feats, attention_mask=mask_encoder)

        base_att = torch.sum(base_att, dim=1)/nums

        ctx_input = torch.cat([base_att, h_gcn, h_lang], 1)

        h_att, c_att = self.att_lstm(ctx_input, (state[0][2], state[1][2]))

        state = (torch.stack([h_lang, h_gcn, h_att]),
                 torch.stack([c_lang, c_gcn, c_att]))

        logprobs = F.log_softmax(self.fc(self.dropout(h_att)), dim=-1)

        return logprobs, state

    def _sample(self, seq, encoder_output, mask_encoder):

        sample_method = 'greedy'
        beam_size = 5
        temperature = 1.0
        sample_n = 5
        group_size = 1
        decoding_constraint = 0
        block_trigrams = 0
        remove_bad_endings = 0
        if beam_size > 1:
            return self._sample_beam(encoder_output, mask_encoder)

        batch_size = encoder_output.size(0)
        state = self.init_hidden(batch_size*sample_n)

        if sample_n > 1:
            p_encoder_output, p_mask_encoder = repeat_tensors(
                sample_n, [encoder_output, mask_encoder])

        trigrams = []  # will be a list of batch_size dictionaries

        seq = encoder_output.new_zeros(
            (batch_size*sample_n, self.seq_length), dtype=torch.long)
        seqLogprobs = encoder_output.new_zeros(
            batch_size*sample_n, self.seq_length, self.vocab_size)
        for t in range(self.seq_length + 1):
            if t == 0:  # input <bos>
                it = encoder_output.new_ones(
                    batch_size*sample_n, dtype=torch.long)*2

            logprobs, state = self.get_logprobs_state(
                it, p_encoder_output, p_mask_encoder, state)

            if decoding_constraint and t > 0:
                tmp = logprobs.new_zeros(logprobs.size())
                tmp.scatter_(1, seq[:, t-1].data.unsqueeze(1), float('-inf'))
                logprobs = logprobs + tmp

            if remove_bad_endings and t > 0:
                tmp = logprobs.new_zeros(logprobs.size())
                prev_bad = np.isin(
                    seq[:, t-1].data.cpu().numpy(), self.bad_endings_ix)
                # Make it impossible to generate bad_endings
                tmp[torch.from_numpy(prev_bad.astype(
                    'uint8')), 0] = float('-inf')
                logprobs = logprobs + tmp

            # Mess with trigrams
            # Copy from https://github.com/lukemelas/image-paragraph-captioning
            if block_trigrams and t >= 3:
                # Store trigram generated at last step
                prev_two_batch = seq[:, t-3:t-1]
                for i in range(batch_size):  # = seq.size(0)
                    prev_two = (prev_two_batch[i][0].item(
                    ), prev_two_batch[i][1].item())
                    current = seq[i][t-1]
                    if t == 3:  # initialize
                        # {LongTensor: list containing 1 int}
                        trigrams.append({prev_two: [current]})
                    elif t > 3:
                        if prev_two in trigrams[i]:  # add to list
                            trigrams[i][prev_two].append(current)
                        else:  # create list
                            trigrams[i][prev_two] = [current]
                # Block used trigrams at next step
                prev_two_batch = seq[:, t-2:t]
                # batch_size x vocab_size
                mask = torch.zeros(logprobs.size(), requires_grad=False).cuda()
                for i in range(batch_size):
                    prev_two = (prev_two_batch[i][0].item(
                    ), prev_two_batch[i][1].item())
                    if prev_two in trigrams[i]:
                        for j in trigrams[i][prev_two]:
                            mask[i, j] += 1
                # Apply mask to log probs
                #logprobs = logprobs - (mask * 1e9)
                alpha = 2.0  # = 4
                # ln(1/2) * alpha (alpha -> infty works best)
                logprobs = logprobs + (mask * -0.693 * alpha)

            # sample the next word
            if t == self.seq_length:  # skip if we achieve maximum length
                break
            it, sampleLogprobs = self.sample_next_word(
                logprobs, sample_method, temperature)

            # stop when all finished
            if t == 0:
                unfinished = it > 0
            else:
                unfinished = unfinished * (it > 0)
            it = it * unfinished.type_as(it)
            seq[:, t] = it
            seqLogprobs[:, t] = logprobs
            # quit loop if all sequences have finished
            if unfinished.sum() == 0:
                break

        return seq, seqLogprobs

    def _sample_beam(self, encoder_output, mask_encoder):
        beam_size = 5
        group_size = 1
        sample_n = 5
        # when sample_n == beam_size then each beam is a sample.
        assert sample_n == 1 or sample_n == beam_size // group_size, 'when beam search, sample_n == 1 or beam search'
        batch_size = encoder_output.size(0)

        assert beam_size <= self.vocab_size
        seq = encoder_output.new_zeros(
            (batch_size*sample_n, self.seq_length), dtype=torch.long)
        seqLogprobs = encoder_output.new_zeros(
            batch_size*sample_n, self.seq_length, self.vocab_size)
        # lets process every image independently for now, for simplicity

        self.done_beams = [[] for _ in range(batch_size)]

        state = self.init_hidden(batch_size)

        # first step, feed bos
        it = encoder_output.new_ones([batch_size], dtype=torch.long)*2
        logprobs, state = self.get_logprobs_state(
            it, encoder_output, mask_encoder, state)

        p_encoder_output, p_mask_encoder = repeat_tensors(
            beam_size, [encoder_output, mask_encoder])
        self.done_beams = self.beam_search(
            state, logprobs, p_encoder_output, p_mask_encoder)
        for k in range(batch_size):
            if sample_n == beam_size:
                for _n in range(sample_n):
                    seq_len = self.done_beams[k][_n]['seq'].shape[0]
                    seq[k*sample_n+_n, :seq_len] = self.done_beams[k][_n]['seq']
                    seqLogprobs[k*sample_n+_n,
                                :seq_len] = self.done_beams[k][_n]['logps']
            else:
                seq_len = self.done_beams[k][0]['seq'].shape[0]
                # the first beam has highest cumulative score
                seq[k, :seq_len] = self.done_beams[k][0]['seq']
                seqLogprobs[k, :seq_len] = self.done_beams[k][0]['logps']
        # return the samples and their log likelihoods
        return seq, seqLogprobs
