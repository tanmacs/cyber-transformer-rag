"""
model.py
========
Full Decoder-only (GPT-style) Transformer Model

Stacks N TransformerBlocks on top of a TokenEmbedding layer.
Final LayerNorm + Linear head projects to vocabulary logits.

Architecture overview:
  token_ids
    ↓
  TokenEmbedding          → [batch, seq_len, d_model]
    ↓
  TransformerBlock × N    → [batch, seq_len, d_model]  (N = 8 layers)
    ↓
  Final LayerNorm         → [batch, seq_len, d_model]
    ↓
  LM Head (Linear)        → [batch, seq_len, vocab_size]

Config used:
  vocab_size  = 50257   (GPT-2 standard, placeholder until Tanmay confirms)
  d_model     = 768     (as specified in task)
  n_layers    = 8       (within required 8–12 range)
  n_heads     = 12      (768 / 12 = 64 per head — clean division)
  d_ffn       = 2048    (standard ~2.7x multiplier for SwiGLU)
  max_seq_len = 512
  dropout     = 0.1
"""

import torch
import torch.nn as nn

from embeddings import TokenEmbedding
from block import TransformerBlock


# ---------------------------------------------------------------------------
# Model Config (dataclass-style for clean passing)
# ---------------------------------------------------------------------------

class TransformerConfig:
    """
    Central config for the Transformer model.
    Change values here — everything else updates automatically.
    """
    vocab_size: int = 559  # Jumaana's tokenizer: 555 tokens + 4 special tokens
    d_model:     int   = 768     # Hidden dimension (FIXED by task spec)
    n_layers:    int   = 8       # Number of stacked decoder blocks
    n_heads:     int   = 12      # Attention heads (768 / 12 = 64 head_dim)
    d_ffn:       int   = 2048    # SwiGLU FFN inner dimension
    max_seq_len: int   = 512     # Max input sequence length
    dropout:     float = 0.1     # Dropout probability


# ---------------------------------------------------------------------------
# Full Transformer Model
# ---------------------------------------------------------------------------

class CyberSecTransformer(nn.Module):
    """
    Decoder-only Transformer for cybersecurity text.

    Design choices:
      - RoPE instead of sinusoidal positional embeddings
      - SwiGLU FFN instead of ReLU/GELU FFN
      - Pre-LayerNorm (more stable training)
      - Causal masking (GPT-style, no encoder)
      - Weight tying: embedding and LM head share weights (reduces params)

    Args:
        config: TransformerConfig instance with all hyperparameters
    """

    def __init__(self, config: TransformerConfig = None):
        super().__init__()

        # Use default config if none provided
        if config is None:
            config = TransformerConfig()

        self.config = config

        # --- Token Embedding ---
        # token_ids [batch, seq_len] → embeddings [batch, seq_len, d_model]
        self.token_embedding = TokenEmbedding(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            dropout=config.dropout,
        )

        # --- Stack of Transformer Blocks ---
        # Each block: [batch, seq_len, d_model] → [batch, seq_len, d_model]
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=config.d_model,
                n_heads=config.n_heads,
                d_ffn=config.d_ffn,
                max_seq_len=config.max_seq_len,
                dropout=config.dropout,
            )
            for _ in range(config.n_layers)
        ])

        # --- Final LayerNorm ---
        # Applied after all blocks, before LM head
        # [batch, seq_len, d_model] → [batch, seq_len, d_model]
        self.final_norm = nn.LayerNorm(config.d_model)

        # --- Language Model Head ---
        # Projects hidden states to vocabulary logits
        # [batch, seq_len, d_model] → [batch, seq_len, vocab_size]
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # --- Weight Tying ---
        # Share weights between token embedding and LM head
        # Standard practice: reduces params, improves training stability
        self.lm_head.weight = self.token_embedding.embedding.weight

        # --- Initialize weights ---
        self._init_weights()

        # Print model size
        total_params = sum(p.numel() for p in self.parameters())
        print(f"CyberSecTransformer initialized.")
        print(f"  Layers      : {config.n_layers}")
        print(f"  d_model     : {config.d_model}")
        print(f"  n_heads     : {config.n_heads}")
        print(f"  head_dim    : {config.d_model // config.n_heads}")
        print(f"  d_ffn       : {config.d_ffn}")
        print(f"  vocab_size  : {config.vocab_size}")
        print(f"  Total params: {total_params:,}")

    def _init_weights(self):
        """
        Initialize weights using standard Transformer init:
          - Linear layers: normal distribution, std=0.02
          - Embedding layers: normal distribution, std=0.02
          - LayerNorm: weight=1, bias=0
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Full forward pass from token IDs to vocabulary logits.

        Args:
            token_ids: [batch, seq_len]  — integer token IDs from tokenizer

        Returns:
            logits: [batch, seq_len, vocab_size]  — raw (unnormalized) scores
                    for next-token prediction at each position
        """
        # --- Token Embedding ---
        # [batch, seq_len] → [batch, seq_len, d_model]
        x = self.token_embedding(token_ids)

        # --- Pass through all Transformer blocks ---
        # Each block: [batch, seq_len, d_model] → [batch, seq_len, d_model]
        for block in self.blocks:
            x = block(x)

        # --- Final LayerNorm ---
        # [batch, seq_len, d_model] → [batch, seq_len, d_model]
        x = self.final_norm(x)

        # --- LM Head ---
        # [batch, seq_len, d_model] → [batch, seq_len, vocab_size]
        logits = self.lm_head(x)

        return logits  # [batch, seq_len, vocab_size]

    def count_parameters(self) -> int:
        """Returns total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
