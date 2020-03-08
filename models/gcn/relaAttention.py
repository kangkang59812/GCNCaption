import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.0):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(3, 4))
        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(
                1).unsqueeze(1) == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output


class RelationMultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        # self.fc = nn.Linear(n_head * d_v, 64, bias=False)
        self.rela_p = nn.Linear(n_head * d_v, 1)
        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        # self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, num_obj, len_q, len_k, len_v = q.size(
            0), q.size(1), q.size(2), k.size(2), v.size(2)

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, num_obj, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, num_obj, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, num_obj, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(2, 3), k.transpose(2, 3), v.transpose(2, 3)

        q = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(2, 3).contiguous().view(sz_b, num_obj, len_q, -1)

        p = self.rela_p(q).squeeze(-1)

        return q, p
