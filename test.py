import unittest
import torch
from unittest.mock import MagicMock

from transformer_common import AbstractModel


class TestAbstractModel(unittest.TestCase):
    def setUp(self):
        # Mock configuration
        self.config = MagicMock()

        # Initialize the model with a mock config
        self.model = AbstractModel(self.config)

        # Define a mock forward method
        def mock_forward(inp):
            # Add 0.1 to all elements
            return inp + 0.1

        # Assign the mock forward to the model
        self.model.forward = mock_forward

    def test_generate(self):
        # Define input tensor (batch_size=1, seq_length=2, features=2)
        inp = torch.tensor([[[0.0, 1.0], [2.0, 3.0]]])  # Shape: (1, 2, 2)

        # Number of new tokens to generate
        max_new_tokens = 1

        # Expected output tensor
        expected_output = torch.tensor([
            [
                [0.0, 1.0],  # Original sequence
                [2.0, 3.0],
                [2.1, 3.1]  # Newly generated tokens (based on mock_forward logic)
            ]
        ])

        # Call the generate method
        output = self.model.generate(inp, max_new_tokens)

        # Print actual and expected outputs for debugging
        print("Actual Output:", output)
        print("Expected Output:", expected_output)

        # Assert the output shape is as expected
        self.assertEqual(output.shape, expected_output.shape)

        # Assert the output values are as expected (with a tolerance)
        self.assertTrue(torch.allclose(output, expected_output, atol=1e-4))


if __name__ == "__main__":
    unittest.main()
