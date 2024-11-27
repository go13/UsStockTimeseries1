import torch
from torch import nn as nn
from torch.nn import functional as F

try:
    from flash_attn.modules.mha import MHA
except ImportError:
    MHA = None

from transformer_common import GeluFeedForward, \
    PositionalEmbedding, DistancePositionalEmbedding, \
    BlockSequence, AbstractModel, TransformerConfig


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
