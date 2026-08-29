import torch
import torch.nn as nn
from typing import cast


"""
## Part 2: RoPE - Rotary Positional Embeddings

**Background:** Instead of adding position information to embeddings, RoPE rotates
the query and key vectors based on their position. This encodes *relative* position
in the attention scores.

**Key insight:** After rotation, the dot product between $q_m$ and $k_n$ depends
only on the distance $(m-n)$, not absolute positions.

**Your task:** Implement RoPE following the steps below.
"""

class RoPE(nn.Module):

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        """
        Args:
            dim: Embedding dimension (must be even)
            max_seq_len: Maximum sequence length
            base: Base for frequency computation
        """
        super().__init__()
        assert dim % 2 == 0, "Dimension must be even"

        tensor_of_indices = torch.arange(0, dim, 2)
        tensor_of_indices = tensor_of_indices.float()
        normalized_tensor_of_indices = tensor_of_indices / dim
        tensor_of_exponents = base ** normalized_tensor_of_indices
        tensor_of_inverse_frequencies = 1.0 / tensor_of_exponents

        tensor_of_positions = torch.arange(0, max_seq_len)
        tensor_of_positions = tensor_of_positions.float()
        tensor_of_frequencies = torch.outer(tensor_of_positions, tensor_of_inverse_frequencies)

        self.register_buffer("tensor_of_frequencies", tensor_of_frequencies)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply rotary embeddings to input.

        Args:
            x: Input of shape (batch, seq_len, dim)
        Returns:
            Rotated input of same shape
        """
        batch, seq_len, dim = x.shape

        # Get frequencies for this sequence length
        buffer = cast(torch.Tensor, self.tensor_of_frequencies)
        tensor_of_frequencies = buffer[:seq_len] # (seq_len, dim/2)

        tensor_of_cosines = torch.cos(tensor_of_frequencies) 
        tensor_of_sines = torch.sin(tensor_of_frequencies)

        # Reshape x into pairs: (batch, seq_len, dim/2, 2)
        x_reshaped = x.reshape(batch, seq_len, -1, 2)

        # Split into even and odd indices
        tensor_of_inputs_with_even_indices = x_reshaped[..., 0]  # (batch, seq_len, dim/2)
        tensor_of_inputs_with_odd_indices = x_reshaped[..., 1]   # (batch, seq_len, dim/2)

        tensor_of_rotated_inputs_with_even_indices = (
            tensor_of_inputs_with_even_indices * tensor_of_cosines - tensor_of_inputs_with_odd_indices * tensor_of_sines
        )

        tensor_of_rotated_inputs_with_odd_indices = (
            tensor_of_inputs_with_even_indices * tensor_of_sines + tensor_of_inputs_with_odd_indices * tensor_of_cosines
        )

        # Stack back together
        tensor_of_rotated_inputs = torch.stack([tensor_of_rotated_inputs_with_even_indices, tensor_of_rotated_inputs_with_odd_indices], dim = -1)

        # Reshape back to original shape
        return tensor_of_rotated_inputs.reshape(batch, seq_len, dim)