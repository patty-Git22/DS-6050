import torch
import torch.nn as nn
from typing import Optional


"""
## Part 4: Complete Transformer Block

Now let's combine everything into a modern Transformer block.
"""

class Transformer(nn.Module):

    def __init__(self, class_RMSNorm, class_MHLA, d_model: int = 256, d_latent: int = 64, mlp_ratio: int = 4):
        """
        Modern Transformer block with:
        - Pre-LN architecture
        - RMSNorm
        - Simplified MHLA
        - Standard MLP

        Args:
            d_model: Model dimension
            d_latent: Latent dimension for MHLA
            mlp_ratio: MLP expansion ratio
        """
        super().__init__()

        # Normalization (Pre-LN style)
        self.RMSNorm_1 = class_RMSNorm(d_model)
        self.RMSNorm_2 = class_RMSNorm(d_model)

        # Attention
        self.MHLA = class_MHLA(d_model, d_latent)

        # MLP
        d_mlp = d_model * mlp_ratio
        self.MLP = nn.Sequential(
            nn.Linear(d_model, d_mlp),
            nn.GELU(),
            nn.Linear(d_mlp, d_model)
        )

    def forward(self, x: torch.Tensor, cache: Optional[torch.Tensor] = None):
        """
        Args:
            x: Input of shape (batch, seq_len, d_model)
            cache: Optional cached L_KV
        Returns:
            output: (batch, seq_len, d_model)
            L_KV: Updated cache
        """

        scaled_normalized_input_tensor_1 = self.RMSNorm_1(x)
        tensor_of_outputs_of_MHLA, cached_tensor_of_latent_inputs = self.MHLA(scaled_normalized_input_tensor_1, cache)
        intermediate = x + tensor_of_outputs_of_MHLA
        scaled_normalized_input_tensor_2 = self.RMSNorm_2(intermediate)
        tensor_of_outputs_of_MLP = self.MLP(scaled_normalized_input_tensor_2)
        tensor_of_outputs = intermediate + tensor_of_outputs_of_MLP

        return tensor_of_outputs, cached_tensor_of_latent_inputs