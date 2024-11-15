import torch
from torch import nn as nn
from torch.nn import functional as F
from flash_attn.modules.mha import MHA

from transformer_common import GeluFeedForward, TransformerConfig,  \
    AbstractRunner
from dataloader import GenericDataloader
def distance_triangle(n, my_device):
    arange_matrix = torch.arange(n, device=my_device).view(-1, 1) - torch.arange(n, device=my_device).view(1, -1)
    lower_triangular = torch.tril(arange_matrix)
    return lower_triangular

class FlashMultiHeadAttention(nn.Module):
    def __init__(self, config: TransformerConfig, causal=True):
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
            causal=causal  # auto-regressive or not
        )


class CausalSelfAttention(nn.Module):
    # https: // pytorch.org / docs / stable / generated / torch.nn.functional.scaled_dot_product_attention.html
    def __init__(self, num_heads: int, embed_dimension: int, bias: bool=False, is_causal: bool=False, dropout:float=0.0):
        super().__init__()
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

    def forward(self, x):
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

class PositionalEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embed)
        self.position_embedding_ff = GeluFeedForward(config.n_embed, config.n_embed, config.n_embed, config.dropout)
        self.position_embedding_ff_ln = nn.LayerNorm(config.n_embed)

    def forward(self, b, t):
        pos_embedding_arrange = torch.arange(t, device=self.config.my_device)
        pos_emb = self.position_embedding_table(pos_embedding_arrange).repeat(b, 1, 1)
        pos_emb = self.position_embedding_ff.forward(pos_emb)
        pos_emb = self.position_embedding_ff_ln(pos_emb)
        return pos_emb


class DistancePositionalEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embed)
        self.position_embedding_ff = GeluFeedForward(
            config.n_embed,
            config.n_embed,
            config.n_embed * 2,
            config.dropout
        )

    def forward(self, b):
        pos_embedding_arrange = distance_triangle(self.config.block_size, self.config.my_device)
        pos_emb = self.position_embedding_table(pos_embedding_arrange)
        pos_emb = pos_emb.repeat(b, 1, 1, 1)
        pos_emb = self.position_embedding_ff.forward(pos_emb)
        return pos_emb


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.sa = FlashMultiHeadAttention(config, causal=config.causal)
        # self.attention = CausalSelfAttention(
        #     num_heads=config.n_head,
        #     embed_dimension=config.n_embed,
        #     bias=False,
        #     is_causal=config.causal,
        #     dropout=config.dropout
        # )

    def forward(self, x, pos_emb, pos_dist_emb):
        return self.attention(x)


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.ffwd = GeluFeedForward(config.n_embed, config.hidden_size, config.n_embed, config.dropout, bias=False)

    def forward(self, x, pos_emb, pos_dist_emb):
        x = x + self.attention(x, pos_emb, pos_dist_emb)
        x = x + self.ffwd(x)
        return x


class BlockSequence(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])

    def forward(self, x, pos_emb, pos_dist_emb):
        for block in self.blocks:
            x = block(x, pos_emb, pos_dist_emb)
        return x


class KarpathyTransformerModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pos_emb1 = PositionalEmbedding(config)
        self.pos_dist_emb1 = DistancePositionalEmbedding(config)
        self.ffwd1 = GeluFeedForward(config.input_embed, config.hidden_size, config.n_embed, config.dropout, bias=False)
        self.t1 = BlockSequence(config)
        self.ffwd2 = GeluFeedForward(config.n_embed, config.hidden_size, config.output_embed, config.dropout, bias=False)

    def forward_vs_target(self, inp, targets):
        output = self.forward(inp)
        b, t, c = output.shape
        logits_view = output.view(b * t, c)
        targets = targets.view(b * t, -1)
        mse_loss = torch.nn.MSELoss(reduction='mean')
        loss = mse_loss(logits_view, targets)
        return output, loss

    def forward(self, inp):
        x = inp
        b, t, c = x.shape
        x = self.ffwd1(x)
        pos_emb = self.pos_emb1(b, t)
        pos_dist_emb = self.pos_dist_emb1(b)
        x = self.t1(x, pos_emb, pos_dist_emb)
        x = self.ffwd2(x)
        return x

    def generate(self, inp, max_new_tokens):
        for _ in range(max_new_tokens):
            x = inp
            x = self.forward(x)
            x = x[-1]
            inp = torch.cat([inp, x.unsqueeze(0)], dim=0)
        return inp


class KarpathyTransformerRunner(AbstractRunner):
    def __init__(self, config, in_data, out_data):
        super().__init__(
            config,
            KarpathyTransformerModel(config),
            GenericDataloader(config, in_data, out_data)
        )
        pass

    @torch.no_grad()
    def generate(self, context, max_new_tokens):
        return self.model.generate(context, max_new_tokens)

