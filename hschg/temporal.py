"""OV-AVEL temporal layer. Active SA path preserved; checkpoint-compatible names.

Adapted from jasongief/OV-AVEL, upstream b5fe1d685d0c6d0d6fd80312b5ccde79f9b73ea6.
The official temporal source credits yujiangpu20/PEL4VAD.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np
import math
from copy import deepcopy as dcp

class DistanceAdj(nn.Module):

    def __init__(self, sigma=0.6, bias=0.2):
        super(DistanceAdj, self).__init__()
        self.w = nn.Parameter(torch.FloatTensor(1))
        self.b = nn.Parameter(torch.FloatTensor(1))
        self.w.data.fill_(sigma)
        self.b.data.fill_(bias)

    def forward(self, batch_size, seq_len, n_heads):
        arith = np.arange(seq_len).reshape(-1, 1)
        dist = torch.from_numpy(np.abs(arith-arith.T).astype(np.float32)).to(self.w.device)
        dist = torch.exp(-torch.abs(self.w * dist ** 2 - self.b))
        dist = dist.unsqueeze(0).unsqueeze(0).repeat(batch_size, n_heads, 1, 1)
        return dist

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, dim_k, dim_v, n_heads, norm_flag=None):
        super(MultiHeadAttention, self).__init__()
        self.dim_v = dim_v
        self.dim_k = dim_k
        self.n_heads = n_heads
        self.norm_flag = norm_flag
        self.q = nn.Linear(d_model, dim_k)
        self.k = nn.Linear(d_model, dim_k)
        self.v = nn.Linear(d_model, dim_v)
        self.o = nn.Linear(dim_v, d_model)
        self.norm_fact = 1 / math.sqrt(dim_k)
        self.act = nn.Softmax(dim=-1)
        self.alpha = nn.Parameter(torch.FloatTensor([0]))

    def forward(self, query, key, value, mask=None, adj=None):
        bs = query.shape[0]
        Q = self.q(query).view(bs, -1, self.n_heads, self.dim_k // self.n_heads).transpose(1, 2)
        K = self.k(key).view(bs, -1, self.n_heads, self.dim_k // self.n_heads).transpose(1, 2)
        V = self.v(value).view(bs, -1, self.n_heads, self.dim_v // self.n_heads).transpose(1, 2)
        if adj is not None:
            g_map = torch.matmul(Q, K.transpose(-2, -1)) * self.norm_fact + adj
        else:
            g_map = torch.matmul(Q, K.transpose(-2, -1)) * self.norm_fact
        if mask is not None:
            l_map = g_map.clone()
            l_map = l_map.masked_fill_(mask.data.eq(0), -1000000000.0)
        g_map = self.act(g_map)
        glb = torch.matmul(g_map, V).transpose(1, 2).reshape(bs, query.shape[1], -1)
        if mask is not None:
            l_map = self.act(l_map)
            lcl = torch.matmul(l_map, V).transpose(1, 2).reshape(bs, query.shape[1], -1)
            alpha = torch.sigmoid(self.alpha)
            tmp = alpha * glb + (1 - alpha) * lcl
        else:
            tmp = glb
        if self.norm_flag:
            tmp = torch.sqrt(F.relu(tmp)) - torch.sqrt(F.relu(-tmp))
            tmp = F.normalize(tmp)
        tmp = self.o(tmp)
        return tmp

class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        output = self.w_2(self.dropout(F.relu(self.w_1(x))))
        return output

class TransformerEncoder(nn.Module):

    def __init__(self, d_model=1024, hid_dim=256, d_ff=512, n_heads=1, dropout=0.1, use_adj_in_attn=False, gamma=0.6, bias=0.2, use_mask_in_attn=False, win_size=1, norm_flag=None):
        super(TransformerEncoder, self).__init__()
        self.n_heads = n_heads
        self.win_size = win_size
        self.self_attn = MultiHeadAttention(d_model, hid_dim, hid_dim, n_heads, norm_flag)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.loc_adj = DistanceAdj(gamma, bias)
        self.use_adj_in_attn = use_adj_in_attn
        self.use_mask_in_attn = use_mask_in_attn

    def forward(self, q, k, v):
        adj = None
        if self.use_adj_in_attn:
            adj = self.loc_adj(q.shape[0], q.shape[1], self.n_heads)
        mask = None
        if self.use_mask_in_attn:
            mask = self.get_mask(self.win_size, q.shape[1], q.shape[0])
        x = q + self.self_attn(q, k, v, mask, adj)
        x = self.norm1(x)
        x = x + self.ffn(x)
        x = self.norm2(x)
        return x

    def get_mask(self, window_size, temporal_scale, bs):
        m = torch.zeros((temporal_scale, temporal_scale))
        w_len = window_size
        for j in range(temporal_scale):
            for k in range(w_len):
                m[j, min(max(j - w_len // 2 + k, 0), temporal_scale - 1)] = 1.0
        m = m.repeat(bs, self.n_heads, 1, 1).cuda()
        return m

class TemporalAttentionModule(nn.Module):

    def __init__(self, sa_layer_num=1, xa_layer_num=1, feat_dim=1024, hid_dim=256, d_ff=512, head_num=1, dropout=0.1, use_adj_in_attn=False, gamma=0.6, bias=0.2, use_mask_in_attn=False, win_size=1, norm_flag=None):
        super(TemporalAttentionModule, self).__init__()
        block = TransformerEncoder(d_model=feat_dim, hid_dim=hid_dim, d_ff=d_ff, n_heads=head_num, dropout=dropout, use_adj_in_attn=use_adj_in_attn, gamma=gamma, bias=bias, use_mask_in_attn=use_mask_in_attn, win_size=win_size, norm_flag=norm_flag)
        self.a_sa_blocks = nn.Sequential(*[dcp(block) for i in range(sa_layer_num)])
        self.a_xa_blocks = nn.Sequential(*[dcp(block) for i in range(xa_layer_num)])
        self.v_sa_blocks = nn.Sequential(*[dcp(block) for i in range(sa_layer_num)])
        self.v_xa_blocks = nn.Sequential(*[dcp(block) for i in range(xa_layer_num)])

    def forward(self, a_tokens, v_tokens):
        for (blk_id, blk) in enumerate(self.a_sa_blocks):
            a_tokens = blk(a_tokens, a_tokens, a_tokens)
        a_sa_tokens = a_tokens
        for (blk_id, blk) in enumerate(self.v_sa_blocks):
            v_tokens = blk(v_tokens, v_tokens, v_tokens)
        v_sa_tokens = v_tokens
        a_tokens = a_sa_tokens
        v_tokens = v_sa_tokens
        if torch.sum(a_tokens.isnan()) > 0 or torch.sum(v_tokens.isnan()) > 0:
            raise FloatingPointError('Nonfinite temporal features')
        return (a_tokens, v_tokens)
