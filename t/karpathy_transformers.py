import torch
from torch import nn as nn
from torch.nn import functional as F
try:
    from flash_attn.modules.mha import MHA
except ImportError:
    MHA = None

from transformer_common import GeluFeedForward, TransformerConfig, \
    PositionalEmbedding, DistancePositionalEmbedding, BlockSequence, AbstractModel

class NNAttentionHead(nn.Module):
    def __init__(self, block_size: int, n_embd: int, head_size: int, dropout: float):
        super().__init__()
        # self.pos_em_ff = nn.Linear(n_embd, n_embd, bias=False)
        self.att = GeluFeedForward(n_embd * 4, n_embd, 1, dropout, bias=True)
        # self.att = LinearFeedForward(n_embd * 3, n_embd, 1, dropout)
        # self.att2 = nn.Linear(n_embd * 4, 1, bias=False)

        self.value = nn.Linear(n_embd, head_size, bias=True)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, pos_emb, pos_dist_emb):
        b, t, c = x.shape

        # pos_emb = self.pos_em_ff(pos_emb)
        x1 = pos_emb + x
        # x1 = x #+ pos_emb
        # x1 = torch.cat([pos_emb, x], dim=-1) # (B,T,C * 2)

        x_tmp = x1.unsqueeze(1).repeat(1, t, 1, 1)  # (B,T,C) -> (B,T,T,C)

        k = x_tmp
        q = x_tmp.transpose(1, 2)

        # a2 = torch.cat([k, q, pos_emb], dim=-1) # (B,T,T,C)
        # a2 = torch.cat([k, q], dim=-1)  # (B,T,T,C)
        # a2 = torch.cat([k, q], dim=-1) + pos_dist_emb  # (B,T,T,C)
        a2 = torch.cat([k, q], dim=-1)  # (B,T,T,C)
        a2 = torch.cat([a2, pos_dist_emb], dim=-1)  # (B,T,T,C)
        # a2 = torch.cat([k, q, pos_emb], dim=-1)   # (B,T,T,C)

        a2 = self.att(a2)  # (B,T,T,C * 2) -> (B,T,T,1)

        wei = a2.squeeze(dim=-1) * c ** -0.5

        # compute attention scores ("affinities")
        wei = wei.masked_fill(self.tril[:t, :t] == 0, float('-inf'))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)

        # perform the weighted aggregation of the values
        v = self.value(x)  # (B,T,C)
        out = wei @ v  # (B, T, T) @ (B, T, C) -> (B, T, C)

        out = self.dropout(out)

        return out


class KarpathyMultiHeadAttention(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()

        dropout = config.dropout
        block_size = config.block_size
        n_embed = config.n_embed
        head_size = config.head_size
        n_head = config.n_head

        self.heads = nn.ModuleList([NNAttentionHead(block_size, n_embed, head_size, dropout) for _ in range(n_head)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, st_pos_emb, pos_dist_emb):
        out = torch.cat([h(x, st_pos_emb, pos_dist_emb) for h in self.heads], dim=-1)
        return out


class KarpathyTransformerModel(AbstractModel):

    def __init__(self, config: TransformerConfig):
        super().__init__(config)

        self.pos_emb1 = PositionalEmbedding(config)
        self.pos_dist_emb1 = DistancePositionalEmbedding(config)

        self.ffwd1 = GeluFeedForward(config.input_embed, config.hidden_size, config.n_embed, config.dropout, bias=False)

        self.t1 = BlockSequence(config, lambda: KarpathyMultiHeadAttention(config))

        self.ffwd2 = GeluFeedForward(config.n_embed, config.hidden_size, config.output_embed, config.dropout, bias=False)


    def forward(self, inp):
        x = inp

        b, t, c = x.shape

        x = self.ffwd1(x)

        pos_emb = self.pos_emb1(b, t)
        pos_dist_emb = self.pos_dist_emb1(b)

        x = self.t1(x, pos_emb, pos_dist_emb)

        x = self.ffwd2(x)

        return x

