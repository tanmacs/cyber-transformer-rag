"""
attention.py
============
Masked Multi-Head Self-Attention Block

Features:
  - Causal (decoder-style) masking — token can only attend to past tokens
  - Rotary Positional Embeddings (RoPE) applied to Q and K
  - Dropout on attention weights
  - Residual connection + LayerNorm (Pre-Norm style — more stable)

Dimension flow:
  Input  x         : [batch, seq_len, d_model]
  Q, K, V projections → [batch, seq_len, d_model]
  Reshape to heads  → [batch, n_heads, seq_len, head_dim]
  Attention output  → [batch, n_heads, seq_len, head_dim]
  Concat heads      → [batch, seq_len, d_model]
  Output projection → [batch, seq_len, d_model]
  After residual    → [batch, seq_len, d_model]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from embeddings import RotaryPositionalEmbedding


# ---------------------------------------------------------------------------
# Masked Multi-Head Self-Attention
# ---------------------------------------------------------------------------

class MaskedMultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention with:
      - Causal mask (decoder / GPT-style)
      - RoPE applied to Q and K
      - Dropout on attention scores
      - Residual connection + Pre-LayerNorm

    Args:
        d_model     : total hidden dimension (e.g. 768)
        n_heads     : number of attention heads (e.g. 12)
        max_seq_len : max sequence length for RoPE cache (e.g. 512)
        dropout     : dropout probability (e.g. 0.1)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()

        assert d_model % n_heads == 0, (
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        )

        self.d_model   = d_model           # 768
        self.n_heads   = n_heads           # 12
        self.head_dim  = d_model // n_heads  # 64

        # Q, K, V projection matrices
        # Each maps [batch, seq_len, d_model] → [batch, seq_len, d_model]
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

        # Output projection: merges all heads back
        # [batch, seq_len, d_model] → [batch, seq_len, d_model]
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        # RoPE for positional encoding on Q and K
        self.rope = RotaryPositionalEmbedding(self.head_dim, max_seq_len)

        # Attention dropout (applied to softmax scores)
        self.attn_dropout = nn.Dropout(dropout)

        # Residual dropout (applied after output projection)
        self.resid_dropout = nn.Dropout(dropout)

        # Pre-LayerNorm (applied BEFORE attention, more stable than post-norm)
        self.layer_norm = nn.LayerNorm(d_model)

        # Scaling factor for dot-product attention
        self.scale = math.sqrt(self.head_dim)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Split d_model into n_heads × head_dim

        Input  : [batch, seq_len, d_model]
        Output : [batch, n_heads, seq_len, head_dim]
        """
        batch, seq_len, _ = x.shape
        # Reshape: [batch, seq_len, n_heads, head_dim]
        x = x.view(batch, seq_len, self.n_heads, self.head_dim)
        # Transpose: [batch, n_heads, seq_len, head_dim]
        return x.transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Merge n_heads × head_dim back into d_model

        Input  : [batch, n_heads, seq_len, head_dim]
        Output : [batch, seq_len, d_model]
        """
        batch, _, seq_len, _ = x.shape
        # Transpose back: [batch, seq_len, n_heads, head_dim]
        x = x.transpose(1, 2).contiguous()
        # Reshape: [batch, seq_len, d_model]
        return x.view(batch, seq_len, self.d_model)

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Build upper-triangular causal mask.
        Positions where mask=True are BLOCKED (filled with -inf before softmax).

        Output: [seq_len, seq_len] boolean tensor
        Example for seq_len=4:
          [[F, T, T, T],
           [F, F, T, T],
           [F, F, F, T],
           [F, F, F, F]]
        Token i can only attend to tokens 0..i (not future tokens)
        """
        mask = torch.ones(seq_len, seq_len, device=device, dtype=torch.bool)
        return torch.triu(mask, diagonal=1)  # upper triangle = True = masked

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]  — input from previous layer

        Returns:
            out: [batch, seq_len, d_model] — same shape, attention applied
        """
        # --- Pre-LayerNorm (normalize BEFORE attention) ---
        # [batch, seq_len, d_model]
        residual = x
        x = self.layer_norm(x)

        batch, seq_len, _ = x.shape

        # --- Project to Q, K, V ---
        # Each: [batch, seq_len, d_model]
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # --- Split into heads ---
        # Each: [batch, n_heads, seq_len, head_dim]
        Q = self._split_heads(Q)
        K = self._split_heads(K)
        V = self._split_heads(V)

        # --- Apply RoPE to Q and K (NOT V) ---
        # RoPE encodes position via rotation of Q and K vectors
        # Each: [batch, n_heads, seq_len, head_dim]
        Q = self.rope(Q)
        K = self.rope(K)

        # --- Scaled Dot-Product Attention ---
        # scores = (Q @ K^T) / sqrt(head_dim)
        # Q: [batch, n_heads, seq_len, head_dim]
        # K^T: [batch, n_heads, head_dim, seq_len]
        # scores: [batch, n_heads, seq_len, seq_len]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # --- Apply Causal Mask ---
        # Fill future positions with -inf so softmax gives them ~0 weight
        mask = self._causal_mask(seq_len, x.device)  # [seq_len, seq_len]
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        # scores: [batch, n_heads, seq_len, seq_len]

        # --- Softmax + Dropout on attention weights ---
        # [batch, n_heads, seq_len, seq_len]
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # --- Weighted sum of V ---
        # attn_weights: [batch, n_heads, seq_len, seq_len]
        # V:            [batch, n_heads, seq_len, head_dim]
        # context:      [batch, n_heads, seq_len, head_dim]
        context = torch.matmul(attn_weights, V)

        # --- Merge heads ---
        # [batch, n_heads, seq_len, head_dim] → [batch, seq_len, d_model]
        context = self._merge_heads(context)

        # --- Output projection ---
        # [batch, seq_len, d_model] → [batch, seq_len, d_model]
        out = self.W_o(context)
        out = self.resid_dropout(out)

        # --- Residual connection ---
        # [batch, seq_len, d_model]
        return residual + out
