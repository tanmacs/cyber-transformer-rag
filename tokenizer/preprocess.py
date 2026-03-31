"""
Custom IOC Tokenizer - Pre-processing Script
Step 3 of 4 in the custom tokenizer pipeline.

This script provides a convenient interface for tokenizing text while preserving
cybersecurity indicators (IOCs) as atomic units. It implements the complete
pre-processing pipeline: regex-based IOC extraction → BPE tokenization.

Usage:
    from preprocess import tokenize_text, detokenize_tokens
    
    # Simple tokenization
    tokens = tokenize_text("Malware at 192.168.1.1 with hash d41d8cd98f...")
    
    # Or with the tokenizer directly
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file("tokenizer.json")
    tokens = tokenizer.encode("text").tokens
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from tokenizers import Tokenizer
import re
from pretokenizer import COMBINED_PATTERN as IOC_REGEX, pretokenize, get_match_type


class IOCTokenizer:
    """
    Domain-aware tokenizer for cybersecurity text.
    
    Features:
    - Preserves IOCs (CVE IDs, IPs, hashes, paths) as atomic tokens
    - Byte-level BPE with 32K vocabulary
    - Custom special tokens for system context
    - Pre-checks IOC boundaries before tokenization
    """
    
    def __init__(self, tokenizer_path: str = "tokenizer.json"):
        """
        Initialize the tokenizer.
        
        Args:
            tokenizer_path: Path to tokenizer.json file
        """
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.special_tokens = {
            "<|system|>",
            "<|context|>",
            "<|query|>",
            "<|endoftext|>",
        }
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text while preserving IOCs as atomic units.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of token strings
        """
        # Phase 1: Identify IOC boundaries using regex
        ioc_spans = []
        for match in IOC_REGEX.finditer(text):
            ioc_spans.append((match.span(), match.group()))
        
        # Phase 2: Pre-tokenize based on IOC boundaries
        pre_tokens = self._pre_tokenize(text, ioc_spans)
        
        # Phase 3: Encode through BPE, respecting IOC boundaries
        tokens = []
        for token_text, is_ioc in pre_tokens:
            if is_ioc:
                # IOC tokens pass through atomically
                tokens.append(token_text)
            else:
                # Regular text is BPE encoded
                encoded = self.tokenizer.encode(token_text).tokens
                tokens.extend(encoded)
        
        return [t for t in tokens if t]
    
    def _pre_tokenize(
        self, 
        text: str, 
        ioc_spans: List[Tuple[Tuple[int, int], str]]
    ) -> List[Tuple[str, bool]]:
        """
        Split text into IOC and non-IOC regions.
        
        Args:
            text: Input text
            ioc_spans: List of (span, ioc_text) tuples
            
        Returns:
            List of (token_text, is_ioc) tuples
        """
        result = []
        last_end = 0
        
        for (start, end), ioc_text in ioc_spans:
            # Add gap before IOC (non-IOC region)
            gap = text[last_end:start]
            if gap.strip():
                # Split gap on whitespace
                for part in gap.split():
                    if part:
                        result.append((part, False))
            
            # Add IOC as atomic unit
            result.append((ioc_text, True))
            last_end = end
        
        # Add trailing gap
        tail = text[last_end:]
        if tail.strip():
            for part in tail.split():
                if part:
                    result.append((part, False))
        
        return result
    
    def tokenize_text_with_types(self, text: str) -> List[Dict]:
        """
        Tokenize text and include IOC type information.
        
        Args:
            text: Input text
            
        Returns:
            List of dicts with 'token' and 'type' keys
        """
        tokens = self.tokenize(text)
        result = []
        
        for token in tokens:
            ioc_type = get_match_type(token)
            result.append({
                "token": token,
                "type": ioc_type,
                "is_special": token in self.special_tokens
            })
        
        return result
    
    def detokenize(self, tokens: List[str]) -> str:
        """
        Reconstruct text from tokens (approximate).
        
        Note: Detokenization is lossy due to BPE subwords.
        This is primarily useful for debugging.
        
        Args:
            tokens: List of tokens
            
        Returns:
            Reconstructed text (approximate)
        """
        # Try to decode through tokenizer
        try:
            # Map tokens back to IDs and decode
            token_ids = []
            for token in tokens:
                # For unknown or special tokens, try to find ID
                if hasattr(self.tokenizer, 'token_to_id'):
                    tid = self.tokenizer.token_to_id(token)
                    if tid is not None:
                        token_ids.append(tid)
            
            if token_ids:
                return self.tokenizer.decode(token_ids)
        except Exception:
            pass
        
        # Fallback: simple space joining
        return " ".join(tokens)


def main():
    """CLI interface for the tokenizer."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="IOC Tokenizer - Domain-aware cybersecurity tokenization"
    )
    parser.add_argument("text", nargs="?", help="Text to tokenize")
    parser.add_argument("--file", type=str, help="File to tokenize (one line per doc)")
    parser.add_argument("--types", action="store_true", help="Show token types")
    parser.add_argument("--tokenizer", type=str, default="tokenizer.json", 
                       help="Path to tokenizer.json")
    
    args = parser.parse_args()
    
    # Load tokenizer
    try:
        tok = IOCTokenizer(args.tokenizer)
    except FileNotFoundError:
        print(f"Error: Tokenizer file not found: {args.tokenizer}")
        return 1
    
    # Tokenize from text or file
    if args.text:
        tokens = tok.tokenize(args.text)
        
        if args.types:
            results = tok.tokenize_text_with_types(args.text)
            print("\nTokens with types:")
            for item in results:
                mark = "🔒" if "IOC" in item["type"] else "📄"
                print(f"  {mark} {item['token']:30} | {item['type']:15}")
        else:
            print(f"\nTokens ({len(tokens)} total):")
            print(f"  {tokens}")
        
        return 0
    
    elif args.file:
        try:
            with open(args.file, "r", encoding="utf-8") as f:
                for i, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        tokens = tok.tokenize(line)
                        print(f"Line {i}: {tokens}")
        except FileNotFoundError:
            print(f"Error: File not found: {args.file}")
            return 1
        return 0
    
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
