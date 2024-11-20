import torch
from torch import nn as nn
from torch.nn import functional as F
try:
    from flash_attn.modules.mha import MHA
except ImportError:
    MHA = None

from transformer_common import GeluFeedForward, TransformerConfig, \
    PositionalEmbedding, DistancePositionalEmbedding, BlockSequence, AbstractModel



class TorchMultiHeadAttention(nn.Module):
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

    def forward(self, x, pos_emb, pos_dist_emb):
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        query_projected = self.c_attn(x)

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


class TorchTransformerModel(AbstractModel):
    def __init__(self, config):
        super().__init__(config)
        self.pos_emb1 = PositionalEmbedding(config)
        self.pos_dist_emb1 = DistancePositionalEmbedding(config)
        self.ffwd1 = GeluFeedForward(config.input_embed, config.hidden_size, config.n_embed, config.dropout, bias=False)
        self.t1 = BlockSequence(config, lambda: TorchMultiHeadAttention(config))
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


class FlashMultiHeadAttention(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        assert torch.bfloat16 == config.precision, 'only bfloat16 is supported - checked 20 aug 23'

        # MHA + rotary requires flash-attention\csrc\rotary>pip install .
        self.flash_mha = MHA(
            embed_dim=config.n_embed,  # total channels (= num_heads * head_dim)
            num_heads=config.n_head,
            device=config.my_device,
            dtype=config.precision,
            dropout=config.dropout,
            use_flash_attn=True,
            return_residual=True,
            dwconv=True,
            rotary_emb_dim=config.head_size,
            causal=config.causal  # auto-regressive or not
        )

    def forward(self, x, pos_emb, pos_dist_emb):
        out = self.flash_mha(x)[0]
        return out

class FlashTransformerModel(AbstractModel):
    def __init__(self, config):
        super().__init__(config)
        self.pos_emb1 = PositionalEmbedding(config)
        self.pos_dist_emb1 = DistancePositionalEmbedding(config)
        self.ffwd1 = GeluFeedForward(config.input_embed, config.hidden_size, config.n_embed, config.dropout, bias=False)
        self.t1 = BlockSequence(config, lambda: FlashMultiHeadAttention(config))
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


class ConvKarpathyTransformerModel(AbstractModel):
    def __init__(self, config):
        super().__init__(config)
        conv_output1 = 8
        conv_output2 = 4
        self.config = config
        self.conv1d1 = nn.Conv1d(
            in_channels=config.input_embed,
            out_channels=config.input_embed * conv_output1,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=config.input_embed
        )
        self.ffwd0 = GeluFeedForward(
            config.input_embed * conv_output1,
            config.hidden_size,
            config.input_embed * conv_output2,
            config.dropout,
            bias=False
        )
        self.conv1d2 = nn.Conv1d(
            in_channels=config.input_embed * conv_output2,
            out_channels=config.input_embed * conv_output2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=config.input_embed
        )
        self.pos_emb1 = PositionalEmbedding(config)
        self.pos_dist_emb1 = DistancePositionalEmbedding(config)
        self.ffwd1 = GeluFeedForward(
            config.input_embed * conv_output2,
            config.hidden_size,
            config.n_embed,
            config.dropout,
            bias=False
        )
        self.t1 = BlockSequence(config, lambda : TorchMultiHeadAttention(config))
        self.ffwd2 = GeluFeedForward(
            config.n_embed,
            config.hidden_size,
            config.output_embed,
            config.dropout,
            bias=False
        )

    def forward(self, inp):
        x = inp
        b, t, c = x.shape

        # Apply convolution; adjust shape for Conv1d
        x = x.permute(0, 2, 1)  # Convert to (batch_size, channels, seq_len)
        x = self.conv1d1(x)  # Apply convolution
        x = x.permute(0, 2, 1)

        x = self.ffwd0(x)

        x = x.permute(0, 2, 1)
        x = self.conv1d2(x)  # Apply convolution
        x = x.permute(0, 2, 1)  # Convert back to (batch_size, seq_len, channels)
        # x now has shape: (batch_size, seq_len, n_embed)

        x = self.ffwd1(x)

        pos_emb = self.pos_emb1(b, t)
        pos_dist_emb = self.pos_dist_emb1(b)
        x = self.t1(x, pos_emb, pos_dist_emb)
        x = self.ffwd2(x)
        return x


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

