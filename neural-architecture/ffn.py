"""
ffn.py
======
SwiGLU Feed-Forward Network (FFN)

Replaces the standard ReLU/GELU FFN used in original Transformers.
Used in: LLaMA, PaLM, Mistral — modern best practice.

Standard FFN:
  x → Linear → GELU → Linear → x

SwiGLU FFN:
  x → [Linear_gate → Swish, Linear_up] → element-wise multiply → Linear_down → x

Why SwiGLU?
  - Gating mechanism allows model to selectively pass information
  - Empirically outperforms GELU/ReLU on language tasks
  - Same parameter count with better expressiveness

Dimension flow:
  Input  x    : [batch, seq_len, d_model]         (768)
  gate / up   : [batch, seq_len, d_ffn]           (2048)
  gate * up   : [batch, seq_len, d_ffn]           (2048)  ← element-wise
  down        : [batch, seq_len, d_model]          (768)
  After residual: [batch, seq_len, d_model]        (768)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# SwiGLU Feed-Forward Network
# ---------------------------------------------------------------------------

class SwiGLUFFN(nn.Module):
    """
    SwiGLU Feed-Forward Network block.

    Architecture:
        gate_proj: d_model → d_ffn
        up_proj:   d_model → d_ffn
        down_proj: d_ffn   → d_model

        output = down_proj( swish(gate_proj(x)) * up_proj(x) )

    Followed by residual connection + Pre-LayerNorm.

    Args:
        d_model  : input/output hidden dimension (e.g. 768)
        d_ffn    : inner FFN dimension (e.g. 2048)
        dropout  : dropout probability (e.g. 0.1)
    """

    def __init__(self, d_model: int, d_ffn: int, dropout: float = 0.1):
        super().__init__()

        self.d_model = d_model  # 768
        self.d_ffn   = d_ffn    # 2048

        # Gate projection: controls what information passes through
        # [batch, seq_len, d_model] → [batch, seq_len, d_ffn]
        self.gate_proj = nn.Linear(d_model, d_ffn, bias=False)

        # Up projection: actual value stream
        # [batch, seq_len, d_model] → [batch, seq_len, d_ffn]
        self.up_proj = nn.Linear(d_model, d_ffn, bias=False)

        # Down projection: project back to d_model
        # [batch, seq_len, d_ffn] → [batch, seq_len, d_model]
        self.down_proj = nn.Linear(d_ffn, d_model, bias=False)

        # Dropout after down projection
        self.dropout = nn.Dropout(dropout)

        # Pre-LayerNorm (applied BEFORE FFN, residual added AFTER)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]

        Returns:
            out: [batch, seq_len, d_model]
        """
        # --- Pre-LayerNorm ---
        # [batch, seq_len, d_model]
        residual = x
        x = self.layer_norm(x)

        # --- Gate stream: Linear → Swish activation ---
        # swish(x) = x * sigmoid(x)  (F.silu is swish)
        # [batch, seq_len, d_model] → [batch, seq_len, d_ffn]
        gate = F.silu(self.gate_proj(x))

        # --- Up stream: Linear (no activation) ---
        # [batch, seq_len, d_model] → [batch, seq_len, d_ffn]
        up = self.up_proj(x)

        # --- SwiGLU gating: element-wise multiply gate and up ---
        # Both: [batch, seq_len, d_ffn]
        # Result: [batch, seq_len, d_ffn]
        gated = gate * up

        # --- Down projection: back to d_model ---
        # [batch, seq_len, d_ffn] → [batch, seq_len, d_model]
        out = self.down_proj(gated)
        out = self.dropout(out)

        # --- Residual connection ---
        # [batch, seq_len, d_model]
        return residual + out
