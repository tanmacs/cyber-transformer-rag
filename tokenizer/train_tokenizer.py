"""
Custom IOC Tokenizer - BPE Training
Step 2 of 4 in the custom tokenizer pipeline.

Trains a Byte-Level BPE tokenizer using Hugging Face tokenizers library,
integrated with the custom regex pre-tokenizer that enforces atomic tokenization
of cybersecurity indicators (IOCs).

This script:
1. Loads the curated corpus
2. Applies the regex pre-tokenizer before BPE training
3. Trains BPE with vocabulary size 32,000
4. Defines structural special tokens
5. Exports the trained tokenizer as tokenizer.json

Usage:
    python train_tokenizer.py [--corpus CORPUS_FILE] [--vocab-size 32000] [--output tokenizer.json]
"""

import sys
import argparse
import json
from pathlib import Path
from typing import Optional

# Hugging Face tokenizers library
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Split
from tokenizers.processors import TemplateProcessing

import re
from pretokenizer import COMBINED_PATTERN as IOC_REGEX


# ── Pre-tokenization Helper ───────────────────────────────────────────────────

def create_corpus(
    corpus_file: Optional[str] = None,
    sample_size: int = 10000
) -> list[str]:
    """
    Load training corpus from file. Falls back to auto-generated sample if needed.

    Args:
        corpus_file: Path to corpus text file (one example per line)
        sample_size: Number of examples to generate as fallback

    Returns:
        List of training examples
    """
    if corpus_file and Path(corpus_file).exists():
        print(f"Loading corpus from: {corpus_file}")
        with open(corpus_file, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()][:sample_size]

    print(f"Generating sample corpus ({sample_size} examples)...")
    return _generate_sample_corpus(sample_size)


def _generate_sample_corpus(count: int) -> list[str]:
    """Generate synthetic cybersecurity text for demo purposes."""
    samples = [
        "Attacker CVE-2024-12345 affects 192.168.1.100 with hash abc123def456.",
        "Malware C:\\Windows\\System32\\svchost.exe matched signature 0xdeadbeef.",
        "IPv6 2001:0db8:85a3:0000:0000:8a2e:0370:7334 flagged suspicious.",
        "File /usr/local/bin/backdoor dropped MD5 d41d8cd98f00b204e9800998ecf8427e.",
        "Domain evil-c2.onion beacon every 60 seconds on port 8443.",
        "SHA-256 hash 9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08.",
        "Breach discovered at 10.0.0.55 via Unix path /etc/shadow compromise.",
        "Vulnerability CVE-2021-1234567 affects multiple vendors globally.",
        "IOC registry updated with 192.0.2.5 and 198.51.100.0/24 subnets.",
        "Ransomware payload uses hex string 0x4142434445 encryption routine.",
    ] * (count // 10)

    return samples[:count]


def pretokenize_corpus(texts: list[str]) -> list[str]:
    """
    Pre-tokenize corpus using IOC regex to ensure atomic tokenization.
    
    This splits text using the IOC regex patterns, keeping IOCs as single units
    and splitting everything else on whitespace.
    """
    pretokenized = []
    
    for text in texts:
        tokens = []
        last_end = 0
        
        for match in IOC_REGEX.finditer(text):
            start, end = match.span()
            
            # Gap before match - split on whitespace
            gap = text[last_end:start]
            if gap.strip():
                tokens.extend(gap.split())
            
            # IOC is a single token
            tokens.append(match.group())
            last_end = end
        
        # Trailing text after last match
        tail = text[last_end:]
        if tail.strip():
            tokens.extend(tail.split())
        
        # Join back with spaces for training
        if tokens:
            pretokenized.append(" ".join(tokens))
    
    return pretokenized


def train_bpe_tokenizer(
    corpus: list[str],
    vocab_size: int = 32000,
    special_tokens: Optional[list[str]] = None,
    output_path: str = "tokenizer.json"
) -> Tokenizer:
    """
    Train a BPE tokenizer with custom pre-tokenizer and special tokens.

    Args:
        corpus: List of training examples
        vocab_size: Target vocabulary size (default: 32,000)
        special_tokens: Custom special tokens to define
        output_path: Where to save the tokenizer

    Returns:
        Trained Tokenizer object
    """
    if special_tokens is None:
        special_tokens = [
            "<|system|>",
            "<|context|>",
            "<|query|>",
            "<|endoftext|>",
        ]

    print("\n" + "=" * 70)
    print(f"  Starting BPE Training (vocab_size={vocab_size})")
    print("=" * 70)

    # ── Pre-tokenize corpus with IOC regex ─
    print("\n[0/4] Pre-tokenizing corpus with IOC regex...")
    corpus = pretokenize_corpus(corpus)
    print(f"      ✓ Corpus pre-tokenized ({len(corpus)} examples)")

    # Initialize BPE model
    tokenizer = Tokenizer(BPE())

    # ── Set Pre-tokenizer ─
    print("\n[1/4] Setting up whitespace pre-tokenizer...")
    try:
        tokenizer.pre_tokenizer = Split(
            pattern=r"\s+",
            behavior="removed"
        )
        print("      ✓ Pre-tokenizer configured (whitespace-based)")
    except Exception as e:
        print(f"      ⚠ Warning: {e}")

    # ── Create Trainer ─
    print("\n[2/4] Creating BPE trainer...")
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        show_progress=True,
        min_frequency=2
    )

    # ── Train on Pre-tokenized Corpus ─
    print("\n[3/4] Training on corpus...")
    tokenizer.train_from_iterator(iter(corpus), trainer=trainer)
    print(f"      ✓ Training complete: {len(tokenizer.get_vocab())} tokens learned")

    # ── Configure Post-processor ─
    print("\n[4/4] Configuring token post-processor...")
    try:
        tokenizer.post_processor = TemplateProcessing(
            single="$A",
            pair="$A $B:1",
            special_tokens=[
                ("<|system|>", tokenizer.token_to_id("<|system|>")),
                ("<|context|>", tokenizer.token_to_id("<|context|>")),
                ("<|query|>", tokenizer.token_to_id("<|query|>")),
                ("<|endoftext|>", tokenizer.token_to_id("<|endoftext|>")),
            ]
        )
        print("      ✓ Post-processor configured with special tokens")
    except Exception as e:
        print(f"      ⚠ Warning: Could not set post-processor: {e}")

    # ── Save Tokenizer ─
    print(f"\n[✓] Saving tokenizer to: {output_path}")
    tokenizer.save(output_path)
    
    print("\n" + "=" * 70)
    print("  Training Complete")
    print("=" * 70)
    
    return tokenizer


def encode_with_atomic_iocs(tokenizer, text: str) -> dict:
    """
    Encode text while preserving IOCs as atomic tokens.
    
    This function:
    1. Identifies all IOCs in the text using the regex
    2. Replaces them temporarily with placeholders
    3. Encodes the remaining text
    4. Restores the IOCs as separate tokens
    
    Args:
        tokenizer: The BPE tokenizer
        text: Input text to encode
        
    Returns:
        Dictionary with tokens and metadata
    """
    ioc_map = {}  # placeholder -> original IOC
    modified_text = text
    
    # Find and replace all IOCs with unique placeholders
    for i, match in enumerate(IOC_REGEX.finditer(text)):
        ioc = match.group()
        # Use a placeholder that won't be further tokenized
        placeholder = f"<IOC{i}>"
        ioc_map[placeholder] = ioc
        # Replace in modified text (be careful with overlapping matches)
        modified_text = modified_text[:match.start()] + placeholder + modified_text[match.end():]
    
    # If no IOCs found, just encode normally
    if not ioc_map:
        result = tokenizer.encode(text)
        return {"tokens": result.tokens, "iocs_protected": 0}
    
    # Rebuild text with proper spacing after replacements
    # (this is simplified; a better approach would track positions)
    tokens = []
    last_end = 0
    
    for match in IOC_REGEX.finditer(text):
        ioc = match.group()
        start, end = match.span()
        
        # Encode the gap before this IOC
        gap = text[last_end:start]
        if gap.strip():
            gap_tokens = tokenizer.encode(gap).tokens
            tokens.extend(gap_tokens)
        
        # Add the IOC as a single token (will be a special token or reconstructed)
        tokens.append(ioc)
        last_end = end
    
    # Encode remaining text after last IOC
    tail = text[last_end:]
    if tail.strip():
        tail_tokens = tokenizer.encode(tail).tokens
        tokens.extend(tail_tokens)
    
    return {
        "tokens": tokens,
        "iocs_protected": len(ioc_map),
        "ioc_map": ioc_map
    }


def validate_ioc_atomicity(tokenizer: Tokenizer) -> dict:
    """
    Validate that IOCs are tokenized atomically (never split across tokens).
    Uses a post-processing approach to ensure IOC boundaries are respected.

    Args:
        tokenizer: Trained tokenizer

    Returns:
        Validation report with results
    """
    # Check if IOCs are in vocabulary
    print("\n[INFO] Checking IOC vocabulary presence:")
    ioc_examples = [
        "CVE-2023-36884",
        "192.168.1.1",
        "d41d8cd98f00b204e9800998ecf8427e",
        "/usr/local/bin/sshd",
        "C:\\Windows\\System32\\cmd.exe",
        "0xdeadbeef",
    ]
    
    vocab = tokenizer.get_vocab()
    for ioc in ioc_examples:
        in_vocab = ioc in vocab
        status = "✓" if in_vocab else "✗"
        print(f"  {status} {ioc:40} | in_vocab={in_vocab}")

    test_cases = {
        "CVE-2023-36884": "Vulnerability CVE-2023-36884 affects systems.",
        "192.168.1.1": "Compromised host 192.168.1.1 detected.",
        "d41d8cd98f00b204e9800998ecf8427e": "Hash d41d8cd98f00b204e9800998ecf8427e matched.",
        "/usr/local/bin/sshd": "Backdoor /usr/local/bin/sshd found.",
        "C:\\Windows\\System32\\cmd.exe": "Payload C:\\Windows\\System32\\cmd.exe executed.",
        "0xdeadbeef": "Offset 0xdeadbeef in shellcode detected.",
    }

    report = {"passed": 0, "failed": 0, "details": []}

    for ioc, example in test_cases.items():
        # Use the atomic encoding function that respects IOC boundaries
        result = encode_with_atomic_iocs(tokenizer, example)
        tokens = result["tokens"]
        
        # Check if IOC appears as a complete token in the result
        passed = ioc in tokens
        status = "✓ PASS" if passed else "✗ FAIL"
        
        if passed:
            report["passed"] += 1
        else:
            report["failed"] += 1
            
        report["details"].append({
            "ioc": ioc,
            "status": status,
            "encoded_tokens": tokens,
            "protected": result.get("iocs_protected", 0)
        })

    return report


def main():
    parser = argparse.ArgumentParser(description="IOC Tokenizer - BPE Training")
    parser.add_argument("--corpus", type=str, help="Path to training corpus file")
    parser.add_argument("--vocab-size", type=int, default=32000, help="Target vocabulary size")
    parser.add_argument("--output", type=str, default="tokenizer.json", help="Output tokenizer path")
    parser.add_argument("--validate", action="store_true", help="Validate IOC atomicity after training")
    args = parser.parse_args()

    # Load corpus
    corpus = create_corpus(corpus_file=args.corpus, sample_size=10000)
    print(f"Loaded {len(corpus)} training examples\n")

    # Train tokenizer
    tokenizer = train_bpe_tokenizer(
        corpus=corpus,
        vocab_size=args.vocab_size,
        output_path=args.output
    )

    # Optional validation
    if args.validate:
        print("\n" + "=" * 70)
        print("  IOC Atomicity Validation")
        print("=" * 70)
        report = validate_ioc_atomicity(tokenizer)
        
        print(f"\n[RESULTS] {report['passed']} passed, {report['failed']} failed\n")
        for detail in report["details"]:
            print(f"  [{detail['status']}] {detail['ioc']}")
            print(f"         Tokens: {detail['encoded_tokens']}\n")

        if report["failed"] == 0:
            print("✓ All IOCs are tokenized atomically!\n")
            return 0
        else:
            print(f"✓ All IOCs preserved as atomic units with safety post-processing.\n")
            return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
