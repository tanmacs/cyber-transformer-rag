"""
dataset.py
==========
PyTorch Dataset for CyberSecTransformer language model training.

Reads corpus_v1.jsonl, tokenizes every document using the trained
tokenizer, concatenates the resulting token IDs into one long stream
(documents separated by <|endoftext|>), then slices that stream into
fixed-length chunks for next-token-prediction training.

Each Dataset item is a pair:
    input_ids  : token_ids[:-1]   shape [max_seq_len]
    target_ids : token_ids[1:]    shape [max_seq_len]  (shifted left by 1)

Usage:
    from dataset import CyberSecDataset
    from torch.utils.data import DataLoader

    dataset = CyberSecDataset(
        corpus_path="../security-corpus/corpus_v1.jsonl",
        tokenizer_path="../tokenizer/tokenizer.json",
        max_seq_len=512,
    )
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    for input_ids, target_ids in loader:
        ...   # input_ids: [B, 512], target_ids: [B, 512]
"""

import json
from pathlib import Path
from typing import List

import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer


class CyberSecDataset(Dataset):
    """
    Language-model dataset for the CyberSecTransformer.

    The full corpus is concatenated into a single token stream with
    <|endoftext|> (ID 3) as a document separator, then sliced into
    non-overlapping windows of size (max_seq_len + 1).  Each window
    becomes one (input_ids, target_ids) training pair.

    Args:
        corpus_path:    Path to corpus_v1.jsonl — one JSON record per line,
                        each record must contain a ``"text"`` field.
        tokenizer_path: Path to tokenizer.json produced by
                        tokenizer/train_tokenizer.py.
        max_seq_len:    Number of tokens per training sample (default: 512).
    """

    # ID of <|endoftext|> in the trained tokenizer
    EOT_TOKEN_ID: int = 3

    def __init__(
        self,
        corpus_path: str,
        tokenizer_path: str,
        max_seq_len: int = 512,
    ):
        self.max_seq_len = max_seq_len

        # Load the trained BPE tokenizer
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

        # Build one long token stream from the entire corpus
        token_stream: List[int] = []
        corpus_path = Path(corpus_path)

        print(f"Loading corpus: {corpus_path}")
        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                text = record.get("text", "").strip()
                if not text:
                    continue
                ids: List[int] = tokenizer.encode(text).ids
                token_stream.extend(ids)
                # Separate documents with <|endoftext|>
                token_stream.append(self.EOT_TOKEN_ID)

        print(f"  Total tokens in stream : {len(token_stream):,}")

        # Slice into non-overlapping chunks of (max_seq_len + 1) tokens.
        # Each chunk yields one (input, target) pair.
        chunk_size = max_seq_len + 1
        n_complete = len(token_stream) // chunk_size
        if n_complete == 0:
            raise ValueError(
                f"Corpus too small: {len(token_stream)} tokens is not enough "
                f"to form even one chunk of {chunk_size} tokens. "
                f"Reduce --max-seq-len or add more corpus data."
            )

        # Trim to a multiple of chunk_size and reshape into [N, chunk_size]
        token_stream = token_stream[: n_complete * chunk_size]
        self.data = torch.tensor(token_stream, dtype=torch.long).view(
            -1, chunk_size
        )
        print(f"  Training samples       : {len(self.data):,}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        """
        Returns:
            input_ids  (torch.LongTensor): shape [max_seq_len]
            target_ids (torch.LongTensor): shape [max_seq_len], shifted left by 1
        """
        chunk = self.data[idx]           # [max_seq_len + 1]
        input_ids  = chunk[:-1]          # [max_seq_len]
        target_ids = chunk[1:]           # [max_seq_len]  (next-token targets)
        return input_ids, target_ids
