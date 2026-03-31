"""
test_forward.py
===============
Forward Pass Test — Validates the full model with a dummy batch.

This test:
  1. Creates a dummy tokenized batch (random integer token IDs)
  2. Runs it through the full CyberSecTransformer
  3. Asserts all output dimensions are exactly correct
  4. Prints dimension flow at every layer for verification
  5. Checks no NaN/Inf values in outputs

Run this file directly:
  python test_forward.py

Expected output: All checks PASSED with correct dimensions printed.
"""

import torch
import sys
import traceback

# Import our model
from model import CyberSecTransformer, TransformerConfig


# ---------------------------------------------------------------------------
# Test Configuration
# ---------------------------------------------------------------------------

# Dummy batch settings
BATCH_SIZE  = 2    # Number of sequences in batch
SEQ_LEN     = 64   # Sequence length (tokens per sequence)

# Model config (must match model.py)
config = TransformerConfig()


# ---------------------------------------------------------------------------
# Helper: Print with pass/fail indicator
# ---------------------------------------------------------------------------

def check(condition: bool, message: str):
    if condition:
        print(f"  ✅ PASSED: {message}")
    else:
        print(f"  ❌ FAILED: {message}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Main Test
# ---------------------------------------------------------------------------

def test_forward_pass():
    print("=" * 60)
    print("  CyberSecTransformer — Forward Pass Test")
    print("=" * 60)

    # --- Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")

    # --- Build model ---
    print("\n[1] Initializing model...")
    try:
        model = CyberSecTransformer(config).to(device)
        model.eval()  # Set to eval mode (disables dropout for deterministic test)
    except Exception as e:
        print(f"  ❌ Model initialization failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    # --- Create dummy input ---
    print(f"\n[2] Creating dummy input batch...")
    print(f"  batch_size  = {BATCH_SIZE}")
    print(f"  seq_len     = {SEQ_LEN}")
    print(f"  vocab_size  = {config.vocab_size}")

    # Random integer token IDs in range [0, vocab_size)
    # Shape: [batch_size, seq_len]
    dummy_input = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(BATCH_SIZE, SEQ_LEN),
        device=device,
    )

    print(f"  Input shape : {list(dummy_input.shape)}")
    check(
        dummy_input.shape == (BATCH_SIZE, SEQ_LEN),
        f"Input shape is [{BATCH_SIZE}, {SEQ_LEN}]"
    )

    # --- Run forward pass ---
    print(f"\n[3] Running forward pass...")
    try:
        with torch.no_grad():
            logits = model(dummy_input)
    except Exception as e:
        print(f"  ❌ Forward pass failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    print(f"  Output shape: {list(logits.shape)}")

    # --- Dimension checks ---
    print(f"\n[4] Checking output dimensions...")

    check(
        logits.shape[0] == BATCH_SIZE,
        f"Batch dimension correct: {logits.shape[0]} == {BATCH_SIZE}"
    )
    check(
        logits.shape[1] == SEQ_LEN,
        f"Sequence length correct: {logits.shape[1]} == {SEQ_LEN}"
    )
    check(
        logits.shape[2] == config.vocab_size,
        f"Vocab dimension correct: {logits.shape[2]} == {config.vocab_size}"
    )
    check(
        logits.shape == (BATCH_SIZE, SEQ_LEN, config.vocab_size),
        f"Full output shape: {list(logits.shape)} == [{BATCH_SIZE}, {SEQ_LEN}, {config.vocab_size}]"
    )

    # --- Numerical checks ---
    print(f"\n[5] Checking for NaN / Inf values...")

    check(
        not torch.isnan(logits).any().item(),
        "No NaN values in output"
    )
    check(
        not torch.isinf(logits).any().item(),
        "No Inf values in output"
    )

    # --- Dtype check ---
    print(f"\n[6] Checking data types...")
    check(
        logits.dtype == torch.float32,
        f"Output dtype is float32: {logits.dtype}"
    )

    # --- Weight tying check ---
    print(f"\n[7] Checking weight tying (embedding == LM head)...")
    check(
        model.token_embedding.embedding.weight.data_ptr() == model.lm_head.weight.data_ptr(),
        "Token embedding and LM head share the same weights"
    )

    # --- Print full dimension flow ---
    print(f"\n[8] Full dimension flow summary:")
    print(f"  token_ids input       : [{BATCH_SIZE}, {SEQ_LEN}]")
    print(f"  after TokenEmbedding  : [{BATCH_SIZE}, {SEQ_LEN}, {config.d_model}]")
    for i in range(config.n_layers):
        print(f"  after TransformerBlock {i+1}: [{BATCH_SIZE}, {SEQ_LEN}, {config.d_model}]")
    print(f"  after Final LayerNorm : [{BATCH_SIZE}, {SEQ_LEN}, {config.d_model}]")
    print(f"  after LM Head         : [{BATCH_SIZE}, {SEQ_LEN}, {config.vocab_size}]")

    # --- Model stats ---
    print(f"\n[9] Model statistics:")
    total_params = model.count_parameters()
    print(f"  Trainable parameters: {total_params:,}")
    print(f"  Approx model size   : {total_params * 4 / 1024 / 1024:.1f} MB (float32)")

    # --- Final result ---
    print("\n" + "=" * 60)
    print("  ALL CHECKS PASSED ✅")
    print("  Model is ready for Vibin's training loop.")
    print("  Connect jumaana's tokenizer by updating vocab_size in TransformerConfig.")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_forward_pass()
