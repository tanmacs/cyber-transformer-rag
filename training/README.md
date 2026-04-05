# Training — CyberSecTransformer

End-to-end language model training for the CyberSecTransformer.

## Files

| File | Purpose |
|------|---------|
| `dataset.py` | PyTorch `Dataset` — tokenizes corpus and builds training samples |
| `train.py` | Main training script — model, optimizer, loop, checkpointing |
| `requirements.txt` | Python dependencies |
| `checkpoints/` | Saved `.pt` checkpoints + `training_log.json` (created at runtime) |

---

## Quick Start

### 1. Install dependencies

```bash
# From repo root
pip install -r training/requirements.txt
# The neural-architecture/ modules require PyTorch (included above)
# The tokenizer modules require tokenizers==0.13.3 (included above)
```

### 2. Run training (all defaults)

```bash
# From repo root — uses corpus_v1.jsonl and tokenizer/tokenizer.json automatically
python training/train.py
```

### 3. Common customisations

```bash
# Shorter run for a quick sanity check
python training/train.py --epochs 2 --batch-size 4 --log-every 5

# Custom corpus / tokenizer paths
python training/train.py \
    --corpus  path/to/corpus.jsonl \
    --tokenizer path/to/tokenizer.json

# Resume from a checkpoint
python training/train.py --resume training/checkpoints/checkpoint_epoch005_step0xxxx.pt
```

---

## CLI Reference

```
usage: train.py [-h] [--corpus CORPUS] [--tokenizer TOKENIZER]
                [--max-seq-len MAX_SEQ_LEN] [--epochs EPOCHS]
                [--batch-size BATCH_SIZE] [--lr LR]
                [--weight-decay WEIGHT_DECAY] [--grad-clip GRAD_CLIP]
                [--warmup-steps WARMUP_STEPS] [--log-every LOG_EVERY]
                [--save-every SAVE_EVERY] [--checkpoint-dir CHECKPOINT_DIR]
                [--resume RESUME]

data:
  --corpus          Path to corpus JSONL (default: security-corpus/corpus_v1.jsonl)
  --tokenizer       Path to tokenizer.json (default: tokenizer/tokenizer.json)
  --max-seq-len     Token sequence length per sample (default: 512)

training:
  --epochs          Number of training epochs (default: 10)
  --batch-size      Batch size (default: 8)
  --lr              Peak learning rate (default: 3e-4)
  --weight-decay    AdamW weight decay (default: 0.1)
  --grad-clip       Gradient clipping max norm (default: 1.0)
  --warmup-steps    Linear LR warmup steps (default: 100)

checkpointing:
  --log-every       Log metrics every N steps (default: 10)
  --save-every      Save checkpoint every N epochs (default: 5)
  --checkpoint-dir  Directory for checkpoints (default: training/checkpoints/)
  --resume          Path to .pt checkpoint to resume from
```

---

## Training Pipeline

```
corpus_v1.jsonl
      │
      │  tokenizers.Tokenizer.encode()
      ▼
  token stream  ─── <|endoftext|> ───► concat all docs
      │
      │  slice into (max_seq_len + 1) chunks
      ▼
  CyberSecDataset
      │
      │  DataLoader (batch_size=8, shuffle=True)
      ▼
  CyberSecTransformer  (8-layer decoder, 768-dim, RoPE, SwiGLU)
      │
      │  CrossEntropyLoss (next-token prediction)
      ▼
  AdamW  (weight-decay on matrices only, betas=(0.9, 0.95))
      │
      │  Cosine LR with linear warmup
      ▼
  checkpoints/checkpoint_epochXXX_stepYYYYYY.pt
  checkpoints/training_log.json
```

### Key hyperparameters

| Hyperparameter | Default | Notes |
|----------------|---------|-------|
| `lr` | `3e-4` | Peak LR; decays to `3e-5` |
| `warmup_steps` | 100 | Linear warmup from 0 |
| `weight_decay` | 0.1 | Applied to weight matrices only |
| `grad_clip` | 1.0 | Max gradient norm |
| `betas` | (0.9, 0.95) | AdamW momentum parameters |
| `batch_size` | 8 | Reduce if GPU OOM |
| `max_seq_len` | 512 | Matches model's `max_seq_len` |

### Checkpoint format

Each `.pt` file contains:
```python
{
    "epoch":                int,
    "step":                 int,
    "model_state_dict":     dict,
    "optimizer_state_dict": dict,
    "loss":                 float,
    "config": {
        "vocab_size": 559, "d_model": 768, "n_layers": 8,
        "n_heads": 12, "d_ffn": 2048, "max_seq_len": 512, "dropout": 0.1
    }
}
```

### Training log

`checkpoints/training_log.json` is written at the end of training:
```json
[
  {"epoch": 1, "step": 10, "loss": 6.23, "avg_loss": 6.41,
   "perplexity": 610.5, "lr": 3e-4},
  ...
]
```

---

## Loading a Checkpoint for Inference

```python
import sys, torch
from pathlib import Path

sys.path.insert(0, "neural-architecture")
from model import CyberSecTransformer, TransformerConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CyberSecTransformer(TransformerConfig()).to(device)

ckpt = torch.load("training/checkpoints/checkpoint_epoch010_stepXXXXXX.pt",
                  map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# Tokenize input
from tokenizers import Tokenizer
tokenizer = Tokenizer.from_file("tokenizer/tokenizer.json")
ids = tokenizer.encode("CVE-2024-12345 affects").ids
input_ids = torch.tensor([ids], dtype=torch.long, device=device)

with torch.no_grad():
    logits = model(input_ids)   # [1, seq_len, vocab_size]
```
