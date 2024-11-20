import unittest
import torch
from transformer_common import TransformerConfig
from transformers import (
    TorchMultiHeadAttention,
    TorchTransformerModel,
    FlashMultiHeadAttention,
    FlashTransformerModel,
    ConvKarpathyTransformerModel
)

class TestTransformerModels(unittest.TestCase):
    def setUp(self):
        # Initialize TransformerConfig for testing
        self.config = TransformerConfig(
            input_embed=128,
            n_embed=128,
            output_embed=64,
            dropout=0.1,
            n_head=8,
            block_size=16,
            causal=True,
            my_device="cpu",
            precision=torch.bfloat16
        )
        self.batch_size = 4
        self.seq_len = 16
        self.input_tensor = torch.rand(self.batch_size, self.seq_len, self.config.input_embed)

    def test_torch_multihead_attention(self):
        # Test TorchMultiHeadAttention
        model = TorchMultiHeadAttention(self.config)
        pos_emb = torch.rand(self.batch_size, self.seq_len, self.config.n_embed)
        pos_dist_emb = torch.rand(self.batch_size, self.seq_len, self.seq_len, self.config.n_embed)
        output = model.forward(self.input_tensor, pos_emb, pos_dist_emb)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.config.n_embed))

    def test_torch_transformer_model(self):
        # Test TorchTransformerModel
        model = TorchTransformerModel(self.config)
        output = model.forward(self.input_tensor)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.config.output_embed))

    def test_flash_multihead_attention(self):
        # Test FlashMultiHeadAttention
        model = FlashMultiHeadAttention(self.config)
        output = model.forward(self.input_tensor)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.config.n_embed))

    def test_flash_transformer_model(self):
        # Test FlashTransformerModel
        model = FlashTransformerModel(self.config)
        output = model.forward(self.input_tensor)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.config.output_embed))

    def test_conv_karpathy_transformer_model(self):
        # Test ConvKarpathyTransformerModel
        model = ConvKarpathyTransformerModel(self.config)
        output = model.forward(self.input_tensor)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.config.output_embed))


if __name__ == "__main__":
    unittest.main()
