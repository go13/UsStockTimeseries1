import torch
from torch import nn as nn
from torch.nn import functional as F
try:
    from flash_attn.modules.mha import MHA
except ImportError:
    MHA = None

from transformer_common import GeluFeedForward, \
    PositionalEmbedding, DistancePositionalEmbedding, BlockSequence, AbstractModel


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
