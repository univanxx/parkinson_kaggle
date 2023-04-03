import torch
import math
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, emb_dim, att_dim):
        super().__init__()
        self.q_matr = nn.Linear(emb_dim, att_dim)
        self.k_matr = nn.Linear(emb_dim, att_dim)
        self.v_matr = nn.Linear(emb_dim, att_dim)
        self.att_dim = att_dim

    def forward(self, x):
        """
        x - input, shape = (B, seq_len, 3)
        """
        q = self.q_matr(x)
        k = self.k_matr(x)
        v = self.v_matr(x)
        return torch.matmul(F.softmax(torch.matmul(q, torch.swapaxes(k, 1, 2))/math.sqrt(self.att_dim), dim=-1), v)


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, emb_dim, att_dim, seq_size):
        super().__init__()
        self.att_heads = nn.ModuleList([Attention(emb_dim, att_dim) for _ in range(num_heads)])
        self.W = nn.Linear(att_dim*num_heads, emb_dim)
        self.seq_size = seq_size
        self.num_heads = num_heads
        self.att_dim = att_dim

    def forward(self, x):
        res = []
        for module in self.att_heads:
            res.append(torch.unsqueeze(module(x), axis=1))
        return self.W(torch.cat(res, axis=1).view(-1, self.seq_size, self.att_dim*self.num_heads))