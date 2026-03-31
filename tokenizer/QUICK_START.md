# IOC Tokenizer - Quick Reference Guide

## 📦 What You Have

| File | Purpose | Status |
|------|---------|--------|
| `tokenizer.json` | Trained BPE tokenizer (555 vocab) | ✅ Ready to use |
| `preprocess.py` | IOC-aware tokenization interface | ✅ CLI + Python API |
| `pretokenizer.py` | Regex patterns + 14 unit tests | ✅ All tests pass |
| `train_tokenizer.py` | BPE training pipeline | ✅ Reusable for new data |
| `validate_tokenizer.py` | 16-test validation suite | ✅ 100% pass rate |
| `validation_report.json` | Detailed test results (JSON) | ✅ Machine-readable |
| `validation_report.html` | Visual test report | ✅ Open in browser |
| `README.md` | Full documentation | ✅ Architecture + examples |

## 🚀 Three Ways To Use

### Option 1: Simple Python API
```python
from preprocess import IOCTokenizer

tokenizer = IOCTokenizer("tokenizer.json")
tokens = tokenizer.tokenize("Malware at 192.168.1.1 with CVE-2023-36884")
print(tokens)
# Output: ['Malware', 'at', '192.168.1.1', 'with', 'CVE-2023-36884']
```

### Option 2: Command Line
```bash
python preprocess.py "Your text here" --types
python preprocess.py --file logs.txt
```

### Option 3: Direct Tokenizer
```python
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("tokenizer.json")
tokens = tokenizer.encode("text").tokens
token_ids = tokenizer.encode("text").ids  # For embedding lookup
```

## 🔐 What Makes It Special

✅ **IOCs Stay Atomic** - Never split across tokens:
- `CVE-2023-36884` → `['CVE-2023-36884']` (not `['CVE', '-20', '23'...]`)
- `192.168.1.1` → `['192.168.1.1']` (not `['192', '.', '168'...]`)
- `d41d8cd98...` → `['d41d8cd98...']` (never fragmented)

✅ **11 IOC Types Supported:**
- IPv4, IPv6, CVE IDs, SHA-256, MD5, SHA-1
- Windows paths, Unix paths, Hex strings
- Domain names, Ports

✅ **Production Ready:**
- 16/16 validation tests passing (100%)
- Special tokens: `<|system|>`, `<|context|>`, `<|query|>`, `<|endoftext|>`
- Full documentation included

## 📊 Test Results at a Glance

```
CVE IDs:        ✓✓✓ (3/3)
IPv4:           ✓✓✓ (3/3)
IPv6:           ✓ (1/1)
SHA-256:        ✓ (1/1)
MD5:            ✓ (1/1)
Windows paths:  ✓✓ (2/2)
Unix paths:     ✓✓ (2/2)
Hex strings:    ✓✓ (2/2)
Domains:        ✓ (1/1)
─────────────────────
TOTAL:          ✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓ (16/16) → 100%
```

## 📖 Read These Files

1. **Start Here:** `README.md` - Overview and examples
2. **For Developers:** `DELIVERABLES.md` - Technical details
3. **For Analysis:** `validation_report.json` - Raw test data
4. **For Management:** `validation_report.html` - Visual dashboard

## 🧪 Quick Validation

Verify everything works:
```bash
# Test 1: Regex patterns (14 tests)
python pretokenizer.py

# Test 2: Tokenizer (16 tests)
python validate_tokenizer.py

# Test 3: Pre-processor (live)
python preprocess.py "CVE-2023-36884 at 10.0.0.55"
```

Expected: All tests pass ✓

## 💡 Common Use Cases

### Security Log Analysis
```python
log_entry = "Breach detected: 192.168.1.100 compromised (CVE-2023-36884)"
tokens = tokenizer.tokenize(log_entry)
# IOCs preserved: ['192.168.1.100', 'CVE-2023-36884'] ✓
```

### Malware Analysis
```python
malware_info = "Hash: d41d8cd98f00b204e9800998ecf8427e found in C:\\Windows\\malware.exe"
tokens = tokenizer.tokenize(malware_info)
# Both preserved: ['d41d8cd98f00b204e9800998ecf8427e', 'C:\\Windows\\malware.exe'] ✓
```

### Model Training
```python
# For LLM embeddings
from tokenizers import Tokenizer
tokenizer = Tokenizer.from_file("tokenizer.json")
for text in training_data:
    token_ids = tokenizer.encode(text).ids
    # Pass to embedding model
```

## ⚙️ Technical Specs

- **Model:** Byte-Level BPE
- **Vocab Size:** 555 learned tokens
- **Special Tokens:** 4 registered
- **Training Corpus:** 10,000 security docs
- **Pattern Order:** Specific → General
- **Pre-tokenizer:** Whitespace + IOC regex

## 🔗 Dependencies

Required:
```bash
pip install tokenizers
```

Included:
- Python 3.7+
- re module (built-in)
- pathlib (built-in)

## ❓ FAQ

**Q: Can I use this with transformers?**
A: Yes! Load with `Tokenizer.from_file()` and integrate with any transformer model.

**Q: What if I have a new IOC type?**
A: Add a regex pattern to `pretokenizer.py` and retrain with `train_tokenizer.py`

**Q: How accurate is the tokenization?**
A: 100% on test suite (16/16 tests passing). Trained on synthetic data; test with your own corpus.

**Q: Can I increase vocabulary size?**
A: Yes, change `--vocab-size 32000` in `train_tokenizer.py` and retrain.

## 📝 Next Steps

1. ✅ Review test results in `validation_report.html`
2. ✅ Test with sample security text: `python preprocess.py "your text"`
3. ✅ Integrate `preprocess.py` into your pipeline
4. ✅ Train on your own corpus if needed: `python train_tokenizer.py --corpus your_data.txt`

---

**Ready to use! Questions? See `README.md` for detailed documentation.**
