import torch
from torch import nn as nn
from torch.nn import functional as F

from transformer_common import GeluFeedForward, TransformerConfig,  \
    AbstractRunner
from dataloader import GenericDataloader


def distance_triangle(n, my_device):
    arange_matrix = torch.arange(n, device=my_device).view(-1, 1) - torch.arange(n, device=my_device).view(1, -1)
    lower_triangular = torch.tril(arange_matrix)
    return lower_triangular


class PositionalEmbedding(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embed)
        self.position_embedding_ff = GeluFeedForward(config.n_embed, config.n_embed, config.n_embed, config.dropout)
        self.position_embedding_ff_ln = nn.LayerNorm(config.n_embed)

    def forward(self, b, t):
        pos_embedding_arrange = torch.arange(t, device=self.config.my_device)
        pos_emb = self.position_embedding_table(pos_embedding_arrange).repeat(b, 1, 1)  # (B,T,C)
        pos_emb = self.position_embedding_ff.forward(pos_emb)
        pos_emb = self.position_embedding_ff_ln(pos_emb)
        # pos_emb = self.dropout(pos_emb)

        # pos_emb = pos_emb.unsqueeze(1).repeat(1, t, 1, 1)  # (B,T,C) -> (B,T,T,C)
        # k = pos_emb
        # q = pos_emb.transpose(1, 2)
        # pos_emb = torch.cat([k, q], dim=-1)  # (B,T,T,C)

        # return k + q
        return pos_emb


class DistancePositionalEmbedding(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embed)
        self.position_embedding_ff = GeluFeedForward(
            config.n_embed,
            config.n_embed,
            config.n_embed * 2,
            config.dropout
        )
        # self.position_embedding_ff_ln = nn.LayerNorm(config.n_embed * 2)

    def forward(self, b):
        pos_embedding_arrange = distance_triangle(self.config.block_size, self.config.my_device)
        pos_emb = self.position_embedding_table(pos_embedding_arrange)
        pos_emb = pos_emb.repeat(b, 1, 1, 1)  # (B, T, T, C)
        pos_emb = self.position_embedding_ff.forward(pos_emb)
        # pos_emb = self.position_embedding_ff_ln(pos_emb)
        return pos_emb

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

        a2 = self.att.forward(a2)  # (B,T,T,C * 2) -> (B,T,T,1)

        wei = a2.squeeze(dim=-1) * c ** -0.5

        # compute attention scores ("affinities")
        wei = wei.masked_fill(self.tril[:t, :t] == 0, float('-inf'))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B,T,C)
        out = wei @ v  # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, config: TransformerConfig, causal=True):
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


class Block(nn.Module):
    def __init__(self, config: TransformerConfig, causal=True):
        super().__init__()

        self.sa = MultiHeadAttention(config, causal=causal)

        self.ffwd = GeluFeedForward(config.n_embed, config.hidden_size, config.n_embed, config.dropout, bias=False)

    def forward(self, x, pos_emb, pos_dist_emb):
        x = x + self.sa.forward(x, pos_emb, pos_dist_emb)
        x = x + self.ffwd.forward(x)
        return x


class BlockSequence(nn.Module):
    def __init__(self, config: TransformerConfig, causal=True):
        super().__init__()
        self.blocks = nn.Sequential(*[Block(config, causal) for _ in range(config.n_layer)])

    def forward(self, x, pos_emb, pos_dist_emb):
        for block in self.blocks:
            x = block(x, pos_emb, pos_dist_emb)
        return x


class KarpathyTransformerModel(nn.Module):

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        self.pos_emb1 = PositionalEmbedding(config)
        self.pos_dist_emb1 = DistancePositionalEmbedding(config)

        self.ffwd1 = GeluFeedForward(config.input_embed, config.hidden_size, config.n_embed, config.dropout, bias=False)

        self.t1 = BlockSequence(config, causal=config.causal)

        self.ffwd2 = GeluFeedForward(config.n_embed, config.hidden_size, config.output_embed, config.dropout, bias=False)

    def forward_vs_target(self, inp, targets):
        output = self.forward(inp)

        b, t, c = output.shape
        # print(output.shape)

        logits_view = output.view(b * t, c)
        targets = targets.view(b * t, -1)

        mse_loss = torch.nn.MSELoss(reduction='mean')
        loss = mse_loss(logits_view, targets)

        # print(f"loss = {loss.item()}, {logits_view.shape}, {targets.shape}")

        return output, loss

    def forward(self, inp):
        x = inp

        b, t, c = x.shape

        x = self.ffwd1.forward(x)

        pos_emb = self.pos_emb1.forward(b, t)
        pos_dist_emb = self.pos_dist_emb1.forward(b)

        x = self.t1.forward(x, pos_emb, pos_dist_emb)

        x = self.ffwd2.forward(x)

        return x

    def generate(self, inp, max_new_tokens):
        # inp is (T, C) array of time by channels
        for _ in range(max_new_tokens):
            #     add batch dimension
            x = inp
            x = self.forward(x)
            # get the last token
            x = x[-1]
            # add the new token to the input
            inp = torch.cat([inp, x.unsqueeze(0)], dim=0)
        return inp


class KarpathyTransformerRunner(AbstractRunner):
    def __init__(self, config: TransformerConfig, in_data, out_data):
        super().__init__(
            config,
            KarpathyTransformerModel(config),
            GenericDataloader(config, in_data, out_data)
        )
        pass

    @torch.no_grad()
    def generate(self, context, max_new_tokens):
        return self.model.generate(context, max_new_tokens)
