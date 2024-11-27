import torch
from torch import nn as nn
from torch.nn import functional as F
try:
    from flash_attn.modules.mha import MHA
except ImportError:
    MHA = None

from transformer_common import GeluFeedForward, TransformerConfig, \
    PositionalEmbedding, DistancePositionalEmbedding, AbstractModel, RMSNorm



class TorchMultiHeadAttention3(nn.Module):
    # https: // pytorch.org / docs / stable / generated / torch.nn.functional.scaled_dot_product_attention.html
    def __init__(self, config):
        super().__init__()

        num_heads = config.n_head
        embed_dimension = config.n_embed
        is_causal = config.causal
        dropout = config.dropout
        bias = False

        assert embed_dimension % num_heads == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(embed_dimension, 3 * embed_dimension, bias=bias)
        # output projection
        self.c_proj = nn.Linear(embed_dimension, embed_dimension, bias=bias)
        # regularization
        self.dropout = dropout
        self.resid_dropout = nn.Dropout(dropout)
        self.num_heads = num_heads
        self.embed_dimension = embed_dimension
        # Perform causal masking
        self.is_causal = is_causal

    def forward(self, x, pos_emb, pos_dist_emb, c_attn_trans):
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        query_projected = self.c_attn(x)

        c_attn_trans = query_projected + c_attn_trans(query_projected)

        batch_size = query_projected.size(0)
        embed_dim = query_projected.size(2)
        head_dim = embed_dim // (self.num_heads * 3)

        query, key, value = query_projected.chunk(3, -1)
        query = query.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)

        # Apply attention with modified scores
        if self.training:
            dropout = self.dropout
            is_causal = self.is_causal
        else:
            dropout = 0.0
            is_causal = False

        y = F.scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=dropout,
                                           is_causal=is_causal)
        y = y.transpose(1, 2).view(batch_size, -1, self.num_heads * head_dim)

        y = self.resid_dropout(self.c_proj(y))
        return y


class Block3(nn.Module):
    def __init__(self, config: TransformerConfig, attention_provider: lambda: nn.Module):
        super().__init__()
        self.l_norm1 = RMSNorm(config.n_embed)
        self.attention = attention_provider()
        self.l_norm2 = RMSNorm(config.n_embed)
        self.ffwd = GeluFeedForward(config.n_embed, config.hidden_size, config.n_embed, config.dropout, bias=False)

    def forward(self, x, pos_emb, pos_dist_emb, c_attn_trans):
        x = pos_emb + x
        x = x + self.attention(self.l_norm1(x), pos_emb, pos_dist_emb, c_attn_trans)
        x = x + self.ffwd.forward(self.l_norm2(x))
        return x


class BlockSequence3(nn.Module):
    def __init__(self, config: TransformerConfig, attention_provider: lambda: nn.Module):
        super().__init__()
        self.blocks = nn.Sequential(*[Block3(config, attention_provider) for _ in range(config.n_layer)])

    def forward(self, x, pos_emb, pos_dist_emb, c_attn_trans):
        for block in self.blocks:
            x = block(x, pos_emb, pos_dist_emb, c_attn_trans)
        return x

class TorchTransformerModel3(AbstractModel):
    def __init__(self, config):
        super().__init__(config)
        self.pos_emb1 = PositionalEmbedding(config)
        self.pos_dist_emb1 = DistancePositionalEmbedding(config)
        self.c_attn_trans = GeluFeedForward(config.n_embed * 3, config.hidden_size * 3, config.n_embed * 3, config.dropout, bias=True)
        self.ffwd1 = GeluFeedForward(config.input_embed, config.hidden_size, config.n_embed, config.dropout, bias=False)
        self.t1 = BlockSequence3(config, lambda: TorchMultiHeadAttention3(config))
        self.ffwd2 = GeluFeedForward(config.n_embed, config.hidden_size, config.output_embed, config.dropout, bias=False)

    def forward(self, inp):
        x = inp
        b, t, c = x.shape
        x = self.ffwd1(x)
        pos_emb = self.pos_emb1(b, t)
        pos_dist_emb = self.pos_dist_emb1(b)
        x = self.t1(x, pos_emb, pos_dist_emb, self.c_attn_trans)
        x = self.ffwd2(x)
        return x

