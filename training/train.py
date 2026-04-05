"""
train.py
========
Training script for the CyberSecTransformer language model.

End-to-end pipeline:
  1. Load corpus + tokenizer → CyberSecDataset
  2. Build CyberSecTransformer (decoder-only, ~62 M params)
  3. AdamW optimizer with weight-decay separation
  4. Cosine LR schedule with linear warmup
  5. GPT-style cross-entropy loss (next-token prediction)
  6. Gradient clipping
  7. Checkpoint saving + JSON training log

Run from the repository root directory:
    python training/train.py

Common options:
    python training/train.py \\
        --corpus  security-corpus/corpus_v1.jsonl \\
        --tokenizer tokenizer/tokenizer.json \\
        --epochs 10 \\
        --batch-size 8 \\
        --lr 3e-4

All paths default to their standard repo locations so the script
works out-of-the-box without any arguments.
"""

import sys
import math
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Make neural-architecture/ importable (model.py, embeddings.py, etc.)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "neural-architecture"))

from model import CyberSecTransformer, TransformerConfig  # noqa: E402

# dataset.py lives in the same training/ directory
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import CyberSecDataset  # noqa: E402


# ---------------------------------------------------------------------------
# Learning-rate schedule
# ---------------------------------------------------------------------------

def get_lr(
    step: int,
    warmup_steps: int,
    max_steps: int,
    max_lr: float,
    min_lr: float,
) -> float:
    """
    Linear warmup followed by cosine decay.

    Args:
        step:         Current global training step (0-indexed).
        warmup_steps: Steps over which LR rises linearly from 0 to max_lr.
        max_steps:    Total training steps; LR reaches min_lr at this point.
        max_lr:       Peak learning rate.
        min_lr:       Floor learning rate (10 % of max_lr by default).

    Returns:
        Learning rate for the current step.
    """
    if step < warmup_steps:
        # Linear warmup
        return max_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    # Cosine decay
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: CyberSecTransformer,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    loss: float,
    checkpoint_dir: Path,
) -> Path:
    """Save model weights, optimizer state, and metadata to a .pt file."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / f"checkpoint_epoch{epoch:03d}_step{step:06d}.pt"
    cfg = model.config
    torch.save(
        {
            "epoch": epoch,
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
            "config": {
                "vocab_size": cfg.vocab_size,
                "d_model":    cfg.d_model,
                "n_layers":   cfg.n_layers,
                "n_heads":    cfg.n_heads,
                "d_ffn":      cfg.d_ffn,
                "max_seq_len": cfg.max_seq_len,
                "dropout":    cfg.dropout,
            },
        },
        path,
    )
    print(f"  Checkpoint saved → {path}")
    return path


def load_checkpoint(
    path: str,
    model: CyberSecTransformer,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
):
    """
    Restore model + optimizer state from a checkpoint.

    Returns:
        (epoch, step) — training position after the checkpoint.
    """
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    print(f"  Resumed from checkpoint: {path}")
    print(f"    epoch={ckpt['epoch']}, step={ckpt['step']}, loss={ckpt['loss']:.4f}")
    return ckpt["epoch"], ckpt["step"]


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    # --- Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # ── 1. Dataset ────────────────────────────────────────────────────────────
    print("[1] Loading dataset...")
    dataset = CyberSecDataset(
        corpus_path=args.corpus,
        tokenizer_path=args.tokenizer,
        max_seq_len=args.max_seq_len,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    print(f"  Batches per epoch      : {len(loader):,}\n")

    # ── 2. Model ──────────────────────────────────────────────────────────────
    print("[2] Initializing model...")
    config = TransformerConfig()
    model = CyberSecTransformer(config).to(device)
    print()

    # ── 3. Optimizer (AdamW with weight-decay separation) ─────────────────────
    # Apply weight decay only to weight matrices (dim >= 2); skip biases and
    # LayerNorm parameters to match the original GPT training recipe.
    decay_params    = [p for _, p in model.named_parameters() if p.dim() >= 2]
    no_decay_params = [p for _, p in model.named_parameters() if p.dim() <  2]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params,    "weight_decay": args.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=args.lr,
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    # ── 4. Loss ───────────────────────────────────────────────────────────────
    loss_fn = nn.CrossEntropyLoss()

    # ── 5. LR schedule params ─────────────────────────────────────────────────
    total_steps  = len(loader) * args.epochs
    warmup_steps = args.warmup_steps
    min_lr       = args.lr * 0.1

    # ── 6. Resume from checkpoint (optional) ─────────────────────────────────
    start_epoch  = 1
    global_step  = 0
    if args.resume:
        start_epoch, global_step = load_checkpoint(
            args.resume, model, optimizer, device
        )
        start_epoch += 1   # resume from the NEXT epoch

    # ── Summary ───────────────────────────────────────────────────────────────
    print("[3] Training configuration:")
    print(f"  epochs         : {args.epochs}")
    print(f"  batch_size     : {args.batch_size}")
    print(f"  max_seq_len    : {args.max_seq_len}")
    print(f"  lr  (peak)     : {args.lr}")
    print(f"  lr  (min)      : {min_lr:.2e}")
    print(f"  warmup_steps   : {warmup_steps}")
    print(f"  total_steps    : {total_steps:,}")
    print(f"  weight_decay   : {args.weight_decay}")
    print(f"  grad_clip      : {args.grad_clip}")
    print(f"  checkpoint_dir : {args.checkpoint_dir}")
    print()

    # ── 7. Training loop ──────────────────────────────────────────────────────
    print("[4] Starting training...\n")
    checkpoint_dir = Path(args.checkpoint_dir)
    log = []

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_loss    = 0.0
        epoch_batches = 0

        for batch_idx, (input_ids, target_ids) in enumerate(loader):
            input_ids  = input_ids.to(device)    # [B, T]
            target_ids = target_ids.to(device)   # [B, T]

            # Update LR before each step
            lr = get_lr(global_step, warmup_steps, total_steps, args.lr, min_lr)
            for group in optimizer.param_groups:
                group["lr"] = lr

            # Forward pass
            logits = model(input_ids)            # [B, T, vocab_size]

            # Compute cross-entropy loss
            # CrossEntropyLoss expects [N, C] and [N], so flatten batch + time
            B, T, V = logits.shape
            loss = loss_fn(
                logits.view(B * T, V),           # [B*T, vocab_size]
                target_ids.view(B * T),          # [B*T]
            )

            # Backward + gradient clip + optimizer step
            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            epoch_loss    += loss.item()
            epoch_batches += 1
            global_step   += 1

            # Periodic logging
            if global_step % args.log_every == 0:
                avg_loss = epoch_loss / epoch_batches
                perplexity = math.exp(min(avg_loss, 20))   # cap to avoid overflow
                entry = {
                    "epoch":       epoch,
                    "step":        global_step,
                    "loss":        round(loss.item(), 4),
                    "avg_loss":    round(avg_loss, 4),
                    "perplexity":  round(perplexity, 2),
                    "lr":          round(lr, 8),
                }
                log.append(entry)
                print(
                    f"  epoch {epoch:3d} | step {global_step:6d} | "
                    f"loss {loss.item():.4f} | avg {avg_loss:.4f} | "
                    f"ppl {perplexity:.2f} | lr {lr:.2e}"
                )

        # End-of-epoch summary
        avg_epoch_loss = epoch_loss / max(1, epoch_batches)
        perplexity = math.exp(min(avg_epoch_loss, 20))
        print(
            f"\n  ── Epoch {epoch:3d} complete: "
            f"avg_loss={avg_epoch_loss:.4f}  ppl={perplexity:.2f} ──\n"
        )

        # Save checkpoint
        if epoch % args.save_every == 0 or epoch == args.epochs:
            save_checkpoint(
                model, optimizer, epoch, global_step, avg_epoch_loss, checkpoint_dir
            )

    # ── 8. Save training log ──────────────────────────────────────────────────
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_path = checkpoint_dir / "training_log.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"\nTraining log saved → {log_path}")
    print("Training complete ✅")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(
        description="Train the CyberSecTransformer language model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Data ──────────────────────────────────────────────────────────────────
    data = parser.add_argument_group("data")
    data.add_argument(
        "--corpus",
        type=str,
        default=str(repo_root / "security-corpus" / "corpus_v1.jsonl"),
        help="Path to corpus JSONL file",
    )
    data.add_argument(
        "--tokenizer",
        type=str,
        default=str(repo_root / "tokenizer" / "tokenizer.json"),
        help="Path to tokenizer.json",
    )
    data.add_argument(
        "--max-seq-len",
        type=int,
        default=512,
        help="Token sequence length per training sample",
    )

    # ── Training ──────────────────────────────────────────────────────────────
    tr = parser.add_argument_group("training")
    tr.add_argument("--epochs",       type=int,   default=10,   help="Number of training epochs")
    tr.add_argument("--batch-size",   type=int,   default=8,    help="Batch size")
    tr.add_argument("--lr",           type=float, default=3e-4, help="Peak learning rate")
    tr.add_argument("--weight-decay", type=float, default=0.1,  help="AdamW weight decay")
    tr.add_argument("--grad-clip",    type=float, default=1.0,  help="Gradient clipping max norm (0 = disabled)")
    tr.add_argument("--warmup-steps", type=int,   default=100,  help="Linear LR warmup steps")

    # ── Logging / checkpointing ───────────────────────────────────────────────
    ck = parser.add_argument_group("checkpointing")
    ck.add_argument("--log-every",  type=int, default=10,
                    help="Log metrics every N steps")
    ck.add_argument("--save-every", type=int, default=5,
                    help="Save checkpoint every N epochs")
    ck.add_argument(
        "--checkpoint-dir",
        type=str,
        default=str(repo_root / "training" / "checkpoints"),
        help="Directory for checkpoints and training log",
    )
    ck.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a checkpoint .pt file to resume training from",
    )

    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
