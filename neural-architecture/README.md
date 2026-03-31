# 🧠 Neural Architecture — Track 1B

**Assigned to:** Charan  
**Track:** 1B — Neural Architecture  
**Project:** Cyber Transformer RAG  

---

## 📌 What This Track Does

This folder contains the **core Transformer model** — the brain of the entire system.

It takes tokenized text (from Jumaana's tokenizer) and learns to understand cybersecurity language.  
The trained model will be used by Aaron's RAG system to generate intelligent responses about threats, CVEs, and attack patterns.

---

## 🏗️ Architecture Overview

```
Token IDs [batch, seq_len]
        ↓
Token Embedding          →  [batch, seq_len, 768]
        ↓
TransformerBlock × 8     →  [batch, seq_len, 768]
  ├── Masked Multi-Head Attention (RoPE)
  └── SwiGLU Feed-Forward Network
        ↓
Final LayerNorm          →  [batch, seq_len, 768]
        ↓
LM Head                  →  [batch, seq_len, vocab_size]
```

---

## 📁 File Structure

| File | What it does |
|---|---|
| `embeddings.py` | Token Embedding layer + Rotary Positional Embeddings (RoPE) |
| `attention.py` | Masked Multi-Head Self-Attention with causal mask + dropout |
| `ffn.py` | SwiGLU Feed-Forward Network (replaces standard ReLU/GELU) |
| `block.py` | One complete Transformer decoder block (Attention + FFN) |
| `model.py` | Full model — stacks 8 blocks, config, weight init |
| `test_forward.py` | Forward pass test — validates all dimensions with dummy batch |

---

## ⚙️ Model Configuration

| Parameter | Value | Reason |
|---|---|---|
| Hidden dimension | 768 | Specified in task |
| Layers | 8 | Within required 8–12 range |
| Attention heads | 12 | 768 ÷ 12 = 64 per head (clean division) |
| Head dimension | 64 | Industry standard |
| FFN dimension | 2048 | ~2.7x multiplier, standard for SwiGLU |
| Max sequence length | 512 | Suitable for cybersecurity text |
| Dropout | 0.1 | Industry standard |
| Vocab size | 559 | Jumaana's tokenizer output (555 + 4 special tokens) |
| Total parameters | 95,246,592 | ~95M — GPT-2 small equivalent |

---

## 🔑 Key Design Choices

### RoPE (Rotary Positional Embeddings)
- Used instead of standard sinusoidal embeddings (as required)
- Rotates Q and K vectors with position-dependent angles
- Better handles long sequences, used in LLaMA and Mistral

### SwiGLU FFN
- Used instead of standard ReLU/GELU FFN (as required)
- Gating mechanism allows selective information flow
- Empirically outperforms standard FFN on language tasks

### Pre-LayerNorm
- LayerNorm applied BEFORE each sub-layer (not after)
- More stable training, used in modern LLMs

### Causal Masking
- Decoder-only GPT-style architecture
- Each token can only attend to previous tokens (not future)

### Weight Tying
- Token embedding and LM head share the same weights
- Reduces parameters, improves training stability

---

## 🚀 How to Run

### Install dependency
```bash
pip install torch
```

### Run forward pass test
```bash
python test_forward.py
```

### Expected output
```
============================================================
  CyberSecTransformer — Forward Pass Test
============================================================
  ✅ PASSED: Input shape is [2, 64]
  ✅ PASSED: Batch dimension correct
  ✅ PASSED: Sequence length correct
  ✅ PASSED: Vocab dimension correct
  ✅ PASSED: Full output shape correct
  ✅ PASSED: No NaN values in output
  ✅ PASSED: No Inf values in output
  ✅ PASSED: Output dtype is float32
  ✅ PASSED: Token embedding and LM head share the same weights
============================================================
  ALL CHECKS PASSED ✅
  Model is ready for Vibin's training loop.
============================================================
```

---

## 🔗 How This Connects to Other Tracks

| Track | Person | Connection to 1B |
|---|---|---|
| 2B — Tokenizer | Jumaana | Provides `vocab_size` → update in `TransformerConfig` |
| 3 — MLOps & Training | Vibin | Imports `CyberSecTransformer` and `TransformerConfig` from `model.py` |
| 1A — RAG | Aaron | Uses the trained model for generation |
| 2A — Dataset | Tanmay/Team | Feeds data to Jumaana's tokenizer → into this model |

---

## 🔧 Connecting Jumaana's Tokenizer

When Jumaana finalizes her tokenizer, update **one line** in `model.py`:

```python
class TransformerConfig:
    vocab_size: int = 559  # ← update this to Jumaana's final vocab size
```

No other changes needed. Everything else updates automatically.

---

## 📦 For Vibin (Training Loop — Track 3)

Import the model like this:

```python
from model import CyberSecTransformer, TransformerConfig

config = TransformerConfig()
model = CyberSecTransformer(config)

# model accepts: token_ids [batch, seq_len]
# model returns: logits    [batch, seq_len, vocab_size]
```

Model supports:
- BF16 training: `model.to(torch.bfloat16)`
- FlashAttention: replace attention computation in `attention.py`
- Gradient checkpointing: wrap blocks in `torch.utils.checkpoint`

---

*Track 1B — Neural Architecture | Charan*
