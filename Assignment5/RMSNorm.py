"""
## Part 1: RMSNorm - Root Mean Square Normalization

**Background:** LayerNorm centers (zero mean) and scales (unit variance).
RMSNorm only scales, which is simpler and faster while being equally effective in Pre-LN architectures.
**Your task:** Complete the RMSNorm implementation below.
"""


import torch
import torch.nn as nn


class RMSNorm(nn.Module):

    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Args:
            dim: Feature dimension
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.eps = eps

        tensor_of_1s = torch.ones(dim)
        self.scale = nn.Parameter(tensor_of_1s)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, dim)
        Returns:
            Normalized tensor of same shape
        """

        tensor_of_squares = x ** 2
        tensor_of_means = tensor_of_squares.mean(dim=-1, keepdim=True)
        tensor_of_means = tensor_of_means + self.eps
        root_mean_square = torch.sqrt(tensor_of_means)
        normalized_input_tensor = x / root_mean_square
        return self.scale * normalized_input_tensor