# CyberSecTransformer — Complete Training Guide

This document covers every stage of the training pipeline end-to-end: corpus
preparation, tokenization, model architecture, the training loop itself, and a
detailed breakdown of the hardware (GPU / storage / RAM) you need to run it.

---

## Table of Contents

1. [Repository Overview](#1-repository-overview)
2. [Process 1 — Corpus & Data Pipeline](#2-process-1--corpus--data-pipeline)
3. [Process 2 — Tokenizer](#3-process-2--tokenizer)
4. [Process 3 — Neural Architecture](#4-process-3--neural-architecture)
5. [Process 4 — Training Pipeline](#5-process-4--training-pipeline)
6. [Hardware Requirements](#6-hardware-requirements)
7. [Hardware Comparison Table](#7-hardware-comparison-table)
8. [Storage Requirements](#8-storage-requirements)
9. [End-to-End Quick-Start](#9-end-to-end-quick-start)

---

## 1. Repository Overview

```
cyber-transformer-rag/
├── security-corpus/          # Raw data & corpus pipeline
│   ├── corpus_v1.jsonl       # 1,705 cybersecurity documents (1.7 MB)
│   ├── data_pipeline/        # Scripts that built corpus_v1.jsonl
│   └── inputs/  outputs/     # Raw source material
│
├── tokenizer/                # Cybersecurity-aware BPE tokenizer
│   ├── pretokenizer.py       # Step 1 – IOC regex patterns
│   ├── train_tokenizer.py    # Step 2 – BPE training
│   ├── preprocess.py         # Step 3 – pre-processing interface
│   ├── validate_tokenizer.py # Step 4 – test suite
│   └── tokenizer.json        # Trained tokenizer (559-token vocabulary)
│
├── neural-architecture/      # Model definition
│   ├── embeddings.py         # Token embedding + RoPE
│   ├── attention.py          # Masked multi-head self-attention
│   ├── ffn.py                # SwiGLU feed-forward network
│   ├── block.py              # Transformer decoder block
│   └── model.py              # Full CyberSecTransformer + config
│
└── training/                 # Training scripts
    ├── dataset.py            # PyTorch Dataset (tokenise + slice corpus)
    ├── train.py              # Main training loop
    └── requirements.txt      # torch, tokenizers
```

---

## 2. Process 1 — Corpus & Data Pipeline

### What the corpus contains

`security-corpus/corpus_v1.jsonl` holds **1,705 cybersecurity documents** (1.7 MB).
Each line is a JSON object with at minimum a `"text"` field:

```json
{"text": "CVE-2024-12345 affects Windows... SHA-256: 9f86d0..."}
```

The documents cover CVE advisories, incident reports, malware analysis, and
threat intelligence bulletins — text that is rich in Indicators of Compromise
(IOCs) such as CVE IDs, IP addresses, cryptographic hashes, and file paths.

### How the corpus was built

The `security-corpus/data_pipeline/` scripts gathered raw source material from
the `inputs/` directory, cleaned it, and wrote the final JSONL to `outputs/`
before it was committed as `corpus_v1.jsonl`.

### Dataset statistics

| Metric | Value |
|--------|-------|
| Documents | 1,705 |
| File size | 1.7 MB |
| Format | JSONL (one JSON record per line) |
| Token stream length (at seq_len=512) | ~994,816 tokens |
| Training samples (seq_len=512) | 1,932 |

---

## 3. Process 2 — Tokenizer

### Why a custom tokenizer?

Standard tokenizers (GPT-2 BPE, SentencePiece) break apart security
indicators:

| Text | GPT-2 BPE result | Custom result |
|------|-----------------|---------------|
| `CVE-2023-36884` | `['CVE', '-20', '23', '-36', '884']` ❌ | `['CVE-2023-36884']` ✅ |
| `192.168.1.1` | `['192', '.', '168', '.1', '.1']` ❌ | `['192.168.1.1']` ✅ |
| `d41d8cd98f00b204e9800998ecf8427e` | Multiple fragments ❌ | Single token ✅ |

### Tokenizer architecture

The tokenizer is a **Byte-Level BPE** model trained with an IOC-aware
pre-tokenizer that uses regex to mark security indicators as atomic units
before BPE merges are learned.

```
Input text
    │
    ▼  Step 1 – pretokenizer.py
IOC regex scan (11 patterns, order: specific → general)
    │   Patterns: CVE IDs, IPv4/IPv6, SHA-256/SHA-1/MD5,
    │             Windows paths, Unix paths, hex strings,
    │             domain names (.onion / .gov / .ru …), ports
    │
    ▼  Step 2 – train_tokenizer.py
BPE training (target vocab 32,000; 555 subword merges learned)
Corpus pre-tokenized by regex before BPE — IOCs never split
    │
    ▼  Resulting tokenizer.json
559 tokens total:
  ~256 single-byte tokens
  ~299 learned BPE merges (common security words, subwords)
    4 special tokens: <|system|>(0) <|context|>(1) <|query|>(2) <|endoftext|>(3)
```

### Special tokens

| Token | ID | Purpose |
|-------|----|---------|
| `<|system|>` | 0 | System / context boundary |
| `<|context|>` | 1 | IOC context snippet marker |
| `<|query|>` | 2 | Query / search marker |
| `<|endoftext|>` | 3 | Document separator (used by the Dataset) |

### Validation results

The tokenizer was validated against 16 test cases covering all IOC types.
**Pass rate: 100% (16/16).**

| IOC type | Tests | Passed |
|----------|-------|--------|
| CVE IDs | 3 | 3 ✅ |
| IPv4 | 3 | 3 ✅ |
| IPv6 | 1 | 1 ✅ |
| SHA-256 | 1 | 1 ✅ |
| MD5 | 1 | 1 ✅ |
| Windows paths | 2 | 2 ✅ |
| Unix paths | 2 | 2 ✅ |
| Hex strings | 2 | 2 ✅ |
| Domain names | 1 | 1 ✅ |

### Tokenizer file size

`tokenizer/tokenizer.json` is ~25 KB — negligible overhead.

---

## 4. Process 3 — Neural Architecture

### Model family

`CyberSecTransformer` is a **decoder-only (GPT-style) Transformer**, built from
four composable modules:

```
neural-architecture/
  embeddings.py  →  TokenEmbedding + RotaryPositionalEmbedding (RoPE)
  attention.py   →  MaskedMultiHeadAttention
  ffn.py         →  SwiGLUFFN
  block.py       →  TransformerBlock (Attention + FFN)
  model.py       →  CyberSecTransformer (stack of N blocks + LM head)
```

### Configuration (`TransformerConfig`)

| Parameter | Value | Notes |
|-----------|-------|-------|
| `vocab_size` | 559 | 555 BPE tokens + 4 special tokens |
| `d_model` | 768 | Hidden dimension (fixed by spec) |
| `n_layers` | 8 | Stacked decoder blocks |
| `n_heads` | 12 | 768 ÷ 12 = 64 per head |
| `d_ffn` | 2048 | SwiGLU inner dimension |
| `max_seq_len` | 512 | Maximum sequence length |
| `dropout` | 0.1 | Applied at embedding, attention, FFN |

### Architecture diagram

```
token_ids  [B, T]
    │
    ▼  TokenEmbedding (vocab_size=559, d_model=768)
    │    • nn.Embedding  [559, 768]
    │    • scale × √d_model
    │    • dropout=0.1
    │
    ▼  [B, T, 768]
    │
    ├─ TransformerBlock × 8  ──────────────────────────────────────────────┐
    │    ┌─ Pre-LayerNorm (768)                                            │
    │    ▼                                                                 │
    │    MaskedMultiHeadAttention                                          │
    │      W_Q, W_K, W_V : Linear(768→768, no bias)                       │
    │      W_O            : Linear(768→768, no bias)                       │
    │      RoPE           : applied to Q and K                             │
    │      Causal mask    : upper-triangular −∞ fill                       │
    │      attn_dropout   : 0.1                                            │
    │      resid_dropout  : 0.1                                            │
    │    + Residual connection                                              │
    │    ┌─ Pre-LayerNorm (768)                                            │
    │    ▼                                                                 │
    │    SwiGLUFFN                                                         │
    │      gate_proj : Linear(768→2048, no bias)                           │
    │      up_proj   : Linear(768→2048, no bias)                           │
    │      down_proj : Linear(2048→768, no bias)                           │
    │      output    = down_proj( silu(gate_proj(x)) * up_proj(x) )       │
    │      dropout   : 0.1                                                 │
    │    + Residual connection                                              │
    └───────────────────────────────────────────────────────────────────────┘
    │
    ▼  Final LayerNorm (768)
    │
    ▼  LM Head — Linear(768→559, no bias)  [weight tied to embedding]
    │
    ▼  logits  [B, T, 559]
```

### Parameter count

| Component | Parameters |
|-----------|------------|
| Token embedding `nn.Embedding(559, 768)` | 429,312 |
| Per block — LayerNorm (×2 per block, weight+bias) | 3,072 |
| Per block — Attention W_Q, W_K, W_V, W_O | 2,359,296 |
| Per block — SwiGLU gate_proj + up_proj + down_proj | 4,718,592 |
| **Per block total** | **7,081,960** |
| **8 blocks total** | **56,655,680** |
| Final LayerNorm | 1,536 |
| LM Head | 0 *(tied to embedding)* |
| **Grand total** | **~57 million** |

Weight tying between the token embedding and LM head is a standard practice
that reduces parameters and improves training stability.

### Design choices vs. GPT-2

| Feature | GPT-2 | CyberSecTransformer |
|---------|-------|---------------------|
| Positional encoding | Learned absolute | **RoPE** (no learned params) |
| FFN activation | GELU | **SwiGLU** (gated, better perf.) |
| Norm placement | Post-LN | **Pre-LN** (more stable) |
| Vocabulary | 50,257 | **559** (domain-specific) |
| Parameters | 117 M (small) | **~57 M** |

---

## 5. Process 4 — Training Pipeline

### Overview

```
corpus_v1.jsonl  (1,705 docs)
       │
       │  CyberSecDataset.__init__()
       │    tokenizer.encode(text).ids  for each doc
       │    append <|endoftext|> (id=3) between docs
       │    → single token stream (~994,816 tokens)
       │    slice into non-overlapping windows of 513 tokens
       │    → 1,932 training samples of shape [513]
       │
       ▼  DataLoader (batch_size=8, shuffle=True, drop_last=True)
       │    → 241 batches per epoch
       │    each batch: input_ids [8,512]  target_ids [8,512]
       │
       ▼  CyberSecTransformer.forward(input_ids)
       │    → logits [8, 512, 559]
       │
       ▼  CrossEntropyLoss
       │    flatten: logits [4096, 559]  targets [4096]
       │    → scalar loss (next-token prediction)
       │
       ▼  loss.backward()
       │    → gradients for all ~57 M parameters
       │
       ▼  nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
       │
       ▼  AdamW.step()
       │    betas=(0.9, 0.95), eps=1e-8
       │    weight_decay=0.1 on weight matrices (dim≥2)
       │    weight_decay=0.0 on biases, LayerNorm params
       │
       ▼  LR schedule (updated every step before forward pass)
       │    steps 0–99    : linear warmup  0 → 3e-4
       │    steps 100–end : cosine decay   3e-4 → 3e-5
       │
       ▼  Checkpoint (saved every 5 epochs and at final epoch)
            training/checkpoints/checkpoint_epochXXX_stepYYYYYY.pt
            training/checkpoints/training_log.json
```

### Dataset construction (dataset.py)

1. **Load tokenizer** — `Tokenizer.from_file("tokenizer/tokenizer.json")`
2. **Tokenize corpus** — iterate over every line of `corpus_v1.jsonl`, call
   `tokenizer.encode(text).ids`, append `<|endoftext|>` (id 3) after each doc.
3. **Build token stream** — a single Python list of ~994,816 integer IDs.
4. **Slice into chunks** — non-overlapping windows of length `max_seq_len + 1`
   (= 513 tokens at the default). Remainder tokens are discarded.
5. **Store as tensor** — `torch.tensor(..., dtype=torch.long).view(-1, 513)`,
   giving a `[1932, 513]` tensor held in CPU RAM.
6. **`__getitem__`** — returns `(chunk[:-1], chunk[1:])` i.e.
   `input_ids [512]` and `target_ids [512]` (shifted left by 1).

### Training loop (train.py)

Step-by-step for every batch:

```
1. input_ids, target_ids = next(loader)    # [B, 512] each
2. move to device (GPU/CPU)
3. lr = get_lr(global_step, ...)           # warmup / cosine
4. set lr on all optimizer param groups
5. logits = model(input_ids)               # forward: [B, 512, 559]
6. loss = CrossEntropyLoss(
       logits.view(B*T, V),                # [4096, 559]
       target_ids.view(B*T)                # [4096]
   )
7. optimizer.zero_grad()
8. loss.backward()
9. clip_grad_norm_(model.parameters(), 1.0)
10. optimizer.step()
11. global_step += 1
12. if global_step % log_every == 0: print metrics
```

End of each epoch:

```
13. compute avg_loss, perplexity = exp(avg_loss)
14. if epoch % save_every == 0 or epoch == final: save checkpoint
```

After all epochs:

```
15. write training_log.json  (list of all logged entries)
```

### Default hyperparameters

| Hyperparameter | Default | CLI flag |
|----------------|---------|----------|
| Epochs | 10 | `--epochs` |
| Batch size | 8 | `--batch-size` |
| Peak learning rate | 3e-4 | `--lr` |
| Min learning rate | 3e-5 *(10 % of peak)* | derived |
| Warmup steps | 100 | `--warmup-steps` |
| Weight decay | 0.1 | `--weight-decay` |
| Gradient clip | 1.0 | `--grad-clip` |
| Max sequence length | 512 | `--max-seq-len` |
| Log every N steps | 10 | `--log-every` |
| Save every N epochs | 5 | `--save-every` |
| AdamW β₁, β₂ | 0.9, 0.95 | hardcoded |
| AdamW ε | 1e-8 | hardcoded |

### Step and epoch counts (defaults)

| Metric | Value |
|--------|-------|
| Samples per epoch | 1,932 |
| Batches per epoch (`batch_size=8, drop_last=True`) | 241 |
| Total steps for 10 epochs | 2,410 |
| Warmup fraction | ~4 % (100 / 2,410) |

### Checkpoint format

Each `.pt` file saved by `save_checkpoint()`:

```python
{
    "epoch":                int,
    "step":                 int,         # global step at save time
    "model_state_dict":     OrderedDict, # all ~57 M parameters
    "optimizer_state_dict": dict,        # AdamW m + v states
    "loss":                 float,       # avg epoch loss at save
    "config": {
        "vocab_size": 559,
        "d_model":    768,
        "n_layers":   8,
        "n_heads":    12,
        "d_ffn":      2048,
        "max_seq_len": 512,
        "dropout":    0.1,
    }
}
```

### Training log format

`training_log.json` is a JSON array, one entry per `log_every` steps:

```json
[
  {
    "epoch": 1,
    "step": 10,
    "loss": 6.23,
    "avg_loss": 6.41,
    "perplexity": 610.5,
    "lr": 0.0003
  },
  ...
]
```

### Resuming training

```bash
python training/train.py \
    --resume training/checkpoints/checkpoint_epoch005_step01205.pt
```

The script restores weights + optimizer state and continues from epoch 6
(start_epoch = saved_epoch + 1), preserving the cosine LR schedule via
`global_step`.

---

## 6. Hardware Requirements

### Memory budget breakdown

During training, the GPU must hold four categories of data simultaneously:

#### A. Model weights (float32)

```
~57,078,528 parameters × 4 bytes = 228 MB
```

#### B. Gradients (float32)

Gradients are the same shape as weights:

```
~57,078,528 gradients × 4 bytes = 228 MB
```

#### C. AdamW optimizer states (float32)

AdamW stores a first-moment (`m`) and second-moment (`v`) tensor for every
parameter:

```
~57,078,528 × 2 states × 4 bytes = 457 MB
```

#### D. Activations (batch-size dependent)

PyTorch stores intermediate activations for backpropagation. The two dominant
contributors per layer are:

- **Attention score matrices** `[B, n_heads, T, T]` = `[B, 12, 512, 512]`
- **Hidden state tensors** `[B, T, D]` = `[B, 512, 768]` (several copies per block)

| Batch size | Attention acts | Hidden acts | Total acts (8 layers) | **Grand total VRAM** |
|-----------|---------------|-------------|----------------------|----------------------|
| B = 2 | ~96 MB | ~96 MB | ~302 MB | **~1.2 GB** |
| B = 4 | ~192 MB | ~192 MB | ~604 MB | **~1.5 GB** |
| B = 8 *(default)* | ~384 MB | ~384 MB | ~1,208 MB | **~2.1 GB** |
| B = 16 | ~768 MB | ~768 MB | ~2,415 MB | **~3.3 GB** |

*(Activation estimates are approximations; exact values depend on PyTorch's
graph allocation.)*

#### Fixed overhead summary

| Component | Memory |
|-----------|--------|
| Model weights (fp32) | 228 MB |
| Gradients (fp32) | 228 MB |
| AdamW m + v states (fp32) | 457 MB |
| **Fixed subtotal** | **913 MB** |
| Activations at B=8 | ~1,208 MB |
| **Total at default B=8** | **~2.1 GB** |

### CPU RAM requirements

The dataset is loaded entirely into CPU RAM as a `[1932, 513]` `torch.LongTensor`:

```
1,932 × 513 × 8 bytes (int64) ≈ 8 MB
```

Total system RAM usage including Python, PyTorch, and tokenizer is typically
**2–4 GB**.

---

## 7. Hardware Comparison Table

The table below compares common GPUs against this training workload.

| GPU | VRAM | Rated TFLOPs (fp32) | B=2 | B=4 | B=8 | B=16 | Notes |
|-----|------|---------------------|-----|-----|-----|------|-------|
| **NVIDIA T4** | 16 GB | 8.1 | ✅ | ✅ | ✅ | ✅ | Google Colab Pro default; great free-tier option |
| **NVIDIA V100 16 GB** | 16 GB | 14.0 | ✅ | ✅ | ✅ | ✅ | Fast; available on AWS p3 |
| **NVIDIA A100 40 GB** | 40 GB | 77.6 | ✅ | ✅ | ✅ | ✅ | Fastest single-GPU option |
| **NVIDIA RTX 4090** | 24 GB | 82.6 | ✅ | ✅ | ✅ | ✅ | Best consumer GPU |
| **NVIDIA RTX 4070** | 12 GB | 40.0 | ✅ | ✅ | ✅ | ✅ | Good mid-range choice |
| **NVIDIA RTX 3070** | 8 GB | 20.3 | ✅ | ✅ | ✅ | ⚠️ ~3.3 GB required | Comfortable up to B=8 |
| **NVIDIA RTX 3060** | 12 GB | 12.7 | ✅ | ✅ | ✅ | ✅ | Good budget GPU |
| **NVIDIA GTX 1080 Ti** | 11 GB | 11.3 | ✅ | ✅ | ✅ | ✅ | Older but sufficient |
| **NVIDIA GTX 1060** | 6 GB | 4.4 | ✅ | ✅ | ⚠️ tight | ❌ OOM | Use B=4 or lower |
| **NVIDIA GTX 1050 Ti** | 4 GB | 2.1 | ✅ | ⚠️ tight | ❌ OOM | B=2 only; very slow |
| **Apple M2 (unified)** | 8–24 GB shared | ~3.6 (GPU) | ✅ | ✅ | ✅ (16 GB model) | MPS backend; slower than discrete CUDA |
| **Apple M1** | 8–16 GB shared | ~2.6 | ✅ | ✅ | ⚠️ (16 GB model) | MPS backend |
| **CPU only (no GPU)** | N/A | N/A | ✅ | ✅ | ✅ | ✅ | Functional; expect 50–200× slower |

**Legend:** ✅ = fits comfortably  ⚠️ = tight / marginal  ❌ = out of memory

### Recommended minimum

| Use case | GPU | Batch size | Expected time (10 epochs) |
|----------|-----|-----------|--------------------------|
| **Just testing** | GTX 1050 Ti (4 GB) or CPU | 2 | Hours (CPU) / ~30 min (GPU) |
| **Development** | RTX 3060 / T4 (12–16 GB) | 8 | ~5–10 min |
| **Production** | RTX 4090 / A100 | 16 | ~2–5 min |

> **Cloud options:**
> - Google Colab (free T4, 15 GB VRAM) — `python training/train.py` runs out-of-the-box
> - Kaggle Notebooks (free T4 / P100)
> - AWS `p3.2xlarge` (V100 16 GB)
> - Lambda Labs GPU instances (A100 / H100)

### Reducing VRAM if OOM

If you hit an out-of-memory error, try these in order:

```bash
# 1 – halve the batch size
python training/train.py --batch-size 4

# 2 – halve again
python training/train.py --batch-size 2

# 3 – shorter sequences (fewer tokens per sample)
python training/train.py --batch-size 4 --max-seq-len 256

# 4 – fall back to CPU (always works, just slow)
# (automatic — train.py uses CPU if no CUDA device is found)
```

---

## 8. Storage Requirements

### File sizes

| File | Size | Notes |
|------|------|-------|
| `security-corpus/corpus_v1.jsonl` | 1.7 MB | Training data |
| `tokenizer/tokenizer.json` | ~25 KB | Trained BPE tokenizer |
| `neural-architecture/*.py` + `training/*.py` | ~50 KB | Source code |
| **1 checkpoint `.pt`** | **~685 MB** | Weights (228 MB) + AdamW states (457 MB) |
| `training_log.json` | ~200 KB | All logged metrics per run |

### Checkpoint storage by save strategy

| Save strategy (`--save-every`) | Epochs | Checkpoints kept | Total disk |
|-------------------------------|--------|-----------------|-----------|
| `--save-every 5` *(default)* | 10 | 2 | **~1.4 GB** |
| `--save-every 2` | 10 | 5 | **~3.4 GB** |
| `--save-every 1` | 10 | 10 | **~6.9 GB** |
| `--save-every 1` | 20 | 20 | **~13.7 GB** |

Checkpoints accumulate — old ones are **not** deleted automatically. Prune
manually if disk is limited.

### Minimum vs. recommended disk

| Scenario | Disk needed |
|----------|------------|
| Minimum (run once, keep 1 checkpoint) | ~2 GB |
| Comfortable (10 epochs, default save-every-5) | ~5 GB |
| Full development (multiple runs, all checkpoints) | 20–50 GB |

---

## 9. End-to-End Quick-Start

### Prerequisites

```bash
# Python 3.8+
pip install -r training/requirements.txt
# installs: torch>=2.0, tokenizers>=0.13.3
```

### Run with all defaults (zero config)

```bash
# From the repository root
python training/train.py
```

This automatically uses:
- `security-corpus/corpus_v1.jsonl` as the corpus
- `tokenizer/tokenizer.json` as the tokenizer
- `training/checkpoints/` as the output directory

### Run with custom settings

```bash
python training/train.py \
    --epochs 20 \
    --batch-size 16 \
    --lr 1e-4 \
    --save-every 2 \
    --log-every 20
```

### Resume interrupted training

```bash
python training/train.py \
    --resume training/checkpoints/checkpoint_epoch005_step01205.pt
```

### Load checkpoint for inference

```python
import sys, torch
from pathlib import Path

sys.path.insert(0, "neural-architecture")
from model import CyberSecTransformer, TransformerConfig
from tokenizers import Tokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Rebuild model
model = CyberSecTransformer(TransformerConfig()).to(device)

# Load weights
ckpt = torch.load(
    "training/checkpoints/checkpoint_epoch010_stepXXXXXX.pt",
    map_location=device,
)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# Tokenize and run
tokenizer = Tokenizer.from_file("tokenizer/tokenizer.json")
ids = tokenizer.encode("CVE-2024-12345 affects").ids
input_ids = torch.tensor([ids], dtype=torch.long, device=device)

with torch.no_grad():
    logits = model(input_ids)   # [1, seq_len, 559]
    next_token = logits[0, -1].argmax().item()
    print("Next token ID:", next_token)
```

---

*Last updated: 2026-04-05*
