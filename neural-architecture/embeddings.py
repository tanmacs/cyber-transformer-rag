"""
embeddings.py
=============
Token Embedding layer + Rotary Positional Embeddings (RoPE)

RoPE rotates query/key vectors using position-dependent angles.
No learned positional parameters — purely mathematical.
Used in: LLaMA, GPT-NeoX, Mistral

Dimension flow:
  Token IDs  : [batch, seq_len]
  Embeddings : [batch, seq_len, d_model]
  RoPE       : applied inside attention on Q/K → [batch, n_heads, seq_len, head_dim]
"""

import torch
import torch.nn as nn
import math


# ---------------------------------------------------------------------------
# 1. Token Embedding
# ---------------------------------------------------------------------------

class TokenEmbedding(nn.Module):
    """
    Converts integer token IDs into dense vectors of size d_model.

    Input  : [batch, seq_len]          (integer token IDs)
    Output : [batch, seq_len, d_model] (float embeddings)
    """

    def __init__(self, vocab_size: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.dropout   = nn.Dropout(dropout)
        self.d_model   = d_model

        # Scale embeddings by sqrt(d_model) — standard transformer trick
        # Prevents embeddings from being too small relative to positional info
        self.scale = math.sqrt(d_model)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: [batch, seq_len] — integer token IDs from tokenizer

        Returns:
            x: [batch, seq_len, d_model] — scaled embedding vectors
        """
        # [batch, seq_len] → [batch, seq_len, d_model]
        x = self.embedding(token_ids) * self.scale
        return self.dropout(x)


# ---------------------------------------------------------------------------
# 2. Rotary Positional Embeddings (RoPE)
# ---------------------------------------------------------------------------

class RotaryPositionalEmbedding(nn.Module):
    """
    RoPE — Rotary Position Embedding (Su et al., 2021)

    Instead of adding positional vectors, RoPE *rotates* the Q and K vectors
    in attention using position-dependent angles. This gives the model
    relative position awareness.

    How it works:
      - Split head_dim into pairs: (x0, x1), (x2, x3), ...
      - Rotate each pair by angle: theta_i * position
      - theta_i = 1 / (10000 ^ (2i / head_dim))

    Applied to: Q and K inside the attention block (NOT to V)

    Dimension flow:
      Input  x : [batch, n_heads, seq_len, head_dim]
      Output x : [batch, n_heads, seq_len, head_dim]  (same shape, rotated)
    """

    def __init__(self, head_dim: int, max_seq_len: int = 512, base: int = 10000):
        super().__init__()
        self.head_dim    = head_dim
        self.max_seq_len = max_seq_len

        # Compute inverse frequencies for each dimension pair
        # theta_i = 1 / (10000 ^ (2i / head_dim))
        # Shape: [head_dim / 2]
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2).float() / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq)

        # Precompute cos/sin cache for all positions up to max_seq_len
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        """Precompute and cache cos/sin values for positions [0, seq_len)"""
        # Positions: [seq_len]
        positions = torch.arange(seq_len, device=self.inv_freq.device).float()

        # Outer product: [seq_len, head_dim/2]
        freqs = torch.outer(positions, self.inv_freq)

        # Concatenate to cover full head_dim: [seq_len, head_dim]
        emb = torch.cat([freqs, freqs], dim=-1)

        # Cache as buffers (not parameters — not trained)
        self.register_buffer("cos_cache", emb.cos())  # [seq_len, head_dim]
        self.register_buffer("sin_cache", emb.sin())  # [seq_len, head_dim]

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """
        Rotate pairs: [x0, x1, x2, x3] → [-x1, x0, -x3, x2]

        Input/Output shape: [batch, n_heads, seq_len, head_dim]
        """
        half = x.shape[-1] // 2
        x1 = x[..., :half]   # first half:  [batch, n_heads, seq_len, head_dim/2]
        x2 = x[..., half:]   # second half: [batch, n_heads, seq_len, head_dim/2]
        return torch.cat([-x2, x1], dim=-1)  # [batch, n_heads, seq_len, head_dim]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RoPE rotation to Q or K tensor.

        Args:
            x: [batch, n_heads, seq_len, head_dim]

        Returns:
            x_rotated: [batch, n_heads, seq_len, head_dim]
        """
        seq_len = x.shape[2]

        # Extend cache if sequence is longer than precomputed
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
            self.max_seq_len = seq_len

        # Slice cos/sin for current sequence length
        # [seq_len, head_dim] → reshape for broadcasting → [1, 1, seq_len, head_dim]
        cos = self.cos_cache[:seq_len].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cache[:seq_len].unsqueeze(0).unsqueeze(0)

        # Apply rotation: x_rot = x * cos + rotate_half(x) * sin
        # [batch, n_heads, seq_len, head_dim]
        return (x * cos) + (self._rotate_half(x) * sin)
