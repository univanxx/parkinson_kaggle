import torch
import math
import torch.nn as nn
import torch.nn.functional as F


class PositionalEmbedding(nn.Module):

    def __init__(self, embed_size, max_len=62):
        super().__init__()

        pe = torch.zeros(max_len+2, embed_size, dtype=torch.float32)
        pe.require_grad = False

        position = torch.arange(0, max_len+2, dtype=torch.float32)

        pe[:, 0] = torch.sin(position / (10000**(2*0 / 3)))
        pe[:, 1] = torch.cos(position / (10000**(2*0 / 3)))
        pe[:, 2] = torch.sin(position / (10000**(2*1 / 3)))

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe

    
class BERTEmbedding(torch.nn.Module):
    """
    BERT Embedding which is consisted with under features
        2. PositionalEmbedding : adding positional information using sin, cos
        sum of all these features are output of BERTEmbedding
    """

    def __init__(self, embed_size=3, seq_len=62):
        """
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """

        super().__init__()
        self.embed_size = embed_size
        # self.pos_embed = nn.Embedding(seq_len, embed_size)
        self.position = PositionalEmbedding(embed_size=embed_size, max_len=seq_len)
        # self.dropout = torch.nn.Dropout(p=dropout)
       
    def forward(self, sequence):
        x = self.position(sequence)
        return x#self.dropout(x)


class Attention(nn.Module):
    def __init__(self, emb_dim, att_dim):
        super().__init__()
        self.q_matr = nn.Linear(emb_dim, att_dim)
        self.k_matr = nn.Linear(emb_dim, att_dim)
        self.v_matr = nn.Linear(emb_dim, att_dim)
        self.att_dim = att_dim

    def forward(self, x, mask=None):
        """
        x - input, shape = (B, seq_len, 3)
        """
        q = self.q_matr(x)
        k = self.k_matr(x)
        v = self.v_matr(x)

        qkmatmul = torch.matmul(q, torch.swapaxes(k, 1, 2))/math.sqrt(self.att_dim)
        if mask is not None:
            qkmatmul = qkmatmul.masked_fill(mask == 0, -1e9)

        return torch.matmul(F.softmax(qkmatmul, dim=-1), v)


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, emb_dim, att_dim, seq_size):
        super().__init__()
        self.att_heads = nn.ModuleList([Attention(emb_dim, att_dim) for _ in range(num_heads)])
        self.W = nn.Linear(att_dim*num_heads, emb_dim)
        self.seq_size = seq_size
        self.num_heads = num_heads
        self.att_dim = att_dim

    def forward(self, x, mask=None):
        res = []
        for module in self.att_heads:
            res.append(module(x, mask))
        return self.W(torch.cat(res, axis=2).view(-1, self.seq_size, self.att_dim*self.num_heads))


class TransformerBlock(nn.Module):
    def __init__(self, num_heads, emb_dim, att_dim, seq_size, hidden_dim):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(num_heads, emb_dim, att_dim, seq_size)
        self.NormalizeFirst = nn.LayerNorm(emb_dim)
        self.FFN = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, emb_dim)
        )
        self.NormalizeLast = nn.LayerNorm(emb_dim)

    def forward(self, x, mask=None):
        x_transformed = self.attention(x, mask)
        x_next = self.NormalizeFirst(x + x_transformed)
        x_transformed = self.FFN(x_next)
        return self.NormalizeLast(x_transformed + x_next)


class NERHead(nn.Module):
    def __init__(self, emb_size, num_classes):
        super(NERHead, self).__init__()
        self.classification_layer = nn.Sequential(
            nn.Linear(emb_size, num_classes),
            nn.Softmax(dim=2)
        )
    def forward(self, x):
        preds = self.classification_layer(x)
        return preds


class BERT4Park(nn.Module):
    def __init__(self, num_blocks=4, num_heads=2, emb_dim=3, att_dim=4, seq_size=62, hidden_dim=4*4, num_classes=4):
        super().__init__()
        self.embedding = BERTEmbedding(emb_dim, seq_size)
        self.transformer_blocks = nn.ModuleList([TransformerBlock(num_heads, emb_dim, att_dim, seq_size+2, hidden_dim) for _ in range(num_blocks)])
        self.classification = NERHead(hidden_dim // 2 + emb_dim, num_classes)
        self.features = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        # self.tokens = nn.ParameterDict({"start": nn.Parameter(torch.rand(emb_dim), requires_grad=True),
        #                "cont": nn.Parameter(torch.rand(emb_dim), requires_grad=True),
        #                "end": nn.Parameter(torch.rand(emb_dim), requires_grad=True)})
    def forward(self, x, features, mask=None):
        # x[(x == torch.tensor([1, 1, 1], dtype=torch.float32).to(device)).all(axis=1).nonzero().flatten()] = self.tokens["start"]
        # x[(x == torch.tensor([2, 2, 2], dtype=torch.float32).to(device)).all(axis=1).nonzero().flatten()] = self.tokens["cont"]
        # x[(x == torch.tensor([-1, -1, -1], dtype=torch.float32).to(device)).all(axis=1).nonzero().flatten()] = self.tokens["end"]
        x = self.embedding(x)
        for transformer in self.transformer_blocks:
            x = transformer(x, mask)
        features = self.features(features)
        x = torch.cat([x,features[:,None,:].repeat(1,x.shape[1],1)],2)
        return self.classification(x)