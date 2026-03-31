"""
block.py
========
Single Transformer Decoder Block

One complete block = Attention + FFN stacked together.
The full model stacks N of these blocks (N = 8 in our config).

Each block follows Pre-Norm style:
  x → LayerNorm → Attention → residual → LayerNorm → FFN → residual

(LayerNorm is inside each sub-module, applied before each sub-layer)

Dimension flow (all tensors stay same shape through entire block):
  Input  : [batch, seq_len, d_model]   (768)
  → Attention block output: [batch, seq_len, d_model]
  → FFN block output:       [batch, seq_len, d_model]
  Output : [batch, seq_len, d_model]   (768)
"""

import torch
import torch.nn as nn

from attention import MaskedMultiHeadAttention
from ffn import SwiGLUFFN


# ---------------------------------------------------------------------------
# Transformer Decoder Block
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """
    One full Transformer decoder block.

    Contains:
      1. MaskedMultiHeadAttention (with RoPE, causal mask, residual, LayerNorm)
      2. SwiGLUFFN               (with residual, LayerNorm)

    Both sub-modules handle their own LayerNorm and residual internally.
    This block just wires them in sequence.

    Args:
        d_model     : hidden dimension (768)
        n_heads     : number of attention heads (12)
        d_ffn       : FFN inner dimension (2048)
        max_seq_len : max sequence length for RoPE (512)
        dropout     : dropout probability (0.1)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ffn: int,
        max_seq_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Sub-layer 1: Masked Multi-Head Self-Attention
        self.attention = MaskedMultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            max_seq_len=max_seq_len,
            dropout=dropout,
        )

        # Sub-layer 2: SwiGLU Feed-Forward Network
        self.ffn = SwiGLUFFN(
            d_model=d_model,
            d_ffn=d_ffn,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]

        Returns:
            x: [batch, seq_len, d_model]  (same shape)
        """
        # Sub-layer 1: Attention (LayerNorm + Attention + Residual inside)
        # [batch, seq_len, d_model] → [batch, seq_len, d_model]
        x = self.attention(x)

        # Sub-layer 2: FFN (LayerNorm + SwiGLU + Residual inside)
        # [batch, seq_len, d_model] → [batch, seq_len, d_model]
        x = self.ffn(x)

        return x  # [batch, seq_len, d_model]
