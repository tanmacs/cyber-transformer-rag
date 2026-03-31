"""
Custom IOC Tokenizer - Regex Pre-Tokenizer
Step 1 of 4 in the custom tokenizer pipeline.

Defines and validates regex patterns that enforce atomic tokenization of
cybersecurity indicators (IOCs). This pre-tokenizer runs BEFORE BPE training
and ensures CVE IDs, IPs, hashes, file paths, and hex strings are never split.

This module is imported by the BPE training script (train_tokenizer.py).
It can also be run standalone to validate all patterns against test cases.

Usage:
    python pretokenizer.py             # runs validation suite
    python pretokenizer.py --verbose   # shows every token split per test
"""

import re
import sys
import argparse
from dataclasses import dataclass

# ── IOC Regex Patterns ────────────────────────────────────────────────────────
#
# ORDER MATTERS — patterns are tried top-to-bottom.
# More specific patterns must come before general ones.
# Each pattern is wrapped in a named group for clarity in debug output.

PATTERNS: list[tuple[str, str]] = [
    # CVE IDs: CVE-2024-12345 or CVE-2024-123456
    ("CVE_ID",          r"CVE-\d{4}-\d{4,7}"),

    # SHA-256: exactly 64 hex chars
    ("SHA256",          r"\b[0-9a-fA-F]{64}\b"),

    # MD5: exactly 32 hex chars
    ("MD5",             r"\b[0-9a-fA-F]{32}\b"),

    # SHA-1: exactly 40 hex chars (between MD5 and SHA256 in specificity)
    ("SHA1",            r"\b[0-9a-fA-F]{40}\b"),

    # IPv6: full and compressed forms
    ("IPV6",            r"\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b"
                        r"|\b(?:[0-9a-fA-F]{1,4}:)*::(?:[0-9a-fA-F]{1,4}:)*[0-9a-fA-F]{1,4}\b"),

    # IPv4: strict octet ranges 0-255
    ("IPV4",            r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}"
                        r"(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"),

    # Windows file paths: C:\Users\foo\bar.exe
    ("WIN_PATH",        r"[A-Za-z]:\\(?:[^\s\\/:*?\"<>|\r\n]+\\)*[^\s\\/:*?\"<>|\r\n]*"),

    # Unix file paths: /usr/bin/python3 or /etc/passwd
    ("UNIX_PATH",       r"(?<!\w)/(?:[^\s/]+/)*[^\s/]+"),

    # Hex strings: 0x prefixed, min 4 chars (e.g. shellcode offsets)
    ("HEX_STRING",      r"\b0x[0-9a-fA-F]{4,}\b"),

    # Domain names with security-relevant TLDs (keep atomic)
    ("DOMAIN",          r"\b(?:[a-zA-Z0-9](?:[a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?\.)+"
                        r"(?:com|net|org|io|gov|mil|edu|ru|cn|onion)\b"),

    # Port numbers in network context: :443 :8080
    ("PORT",            r":\d{1,5}\b"),
]

# Compile into a single alternation regex — order preserved
COMBINED_PATTERN = re.compile(
    "|".join(f"(?P<{name}>{pattern})" for name, pattern in PATTERNS)
)


# ── Pre-tokenizer function ────────────────────────────────────────────────────

def pretokenize(text: str) -> list[str]:
    """
    Split text into tokens, preserving IOCs as atomic units.

    Non-IOC spans are split on whitespace; IOC spans are returned
    as single tokens regardless of their internal structure.

    Args:
        text: Raw input string

    Returns:
        List of token strings
    """
    tokens = []
    last_end = 0

    for match in COMBINED_PATTERN.finditer(text):
        start, end = match.span()

        # Tokenize the gap before this match on whitespace
        gap = text[last_end:start]
        if gap.strip():
            tokens.extend(gap.split())

        # The IOC match is always a single atomic token
        tokens.append(match.group())
        last_end = end

    # Handle any trailing text after the last match
    tail = text[last_end:]
    if tail.strip():
        tokens.extend(tail.split())

    return [t for t in tokens if t]


def get_match_type(token: str) -> str:
    """Return the pattern name that matched a token, or 'text' if none."""
    m = COMBINED_PATTERN.fullmatch(token)
    if m:
        return m.lastgroup
    return "text"


# ── Validation Suite ──────────────────────────────────────────────────────────

@dataclass
class TestCase:
    name:           str
    input:          str
    must_contain:   list[str]   # these tokens must appear exactly as-is
    must_not_split: list[str]   # these strings must never be split across tokens


TEST_CASES: list[TestCase] = [
    TestCase(
        name="CVE ID — standard",
        input="Vulnerability CVE-2024-12345 affects OpenSSL.",
        must_contain=["CVE-2024-12345"],
        must_not_split=["CVE-2024-12345"],
    ),
    TestCase(
        name="CVE ID — long suffix",
        input="Patch released for CVE-2021-1234567 immediately.",
        must_contain=["CVE-2021-1234567"],
        must_not_split=["CVE-2021-1234567"],
    ),
    TestCase(
        name="CVE ID — multiple in sentence",
        input="Both CVE-2023-44487 and CVE-2022-30190 are critical.",
        must_contain=["CVE-2023-44487", "CVE-2022-30190"],
        must_not_split=["CVE-2023-44487", "CVE-2022-30190"],
    ),
    TestCase(
        name="IPv4 — standard",
        input="C2 server located at 192.168.1.105 was blocked.",
        must_contain=["192.168.1.105"],
        must_not_split=["192.168.1.105"],
    ),
    TestCase(
        name="IPv4 — edge octets",
        input="Traffic from 0.0.0.0 and 255.255.255.255 logged.",
        must_contain=["0.0.0.0", "255.255.255.255"],
        must_not_split=["0.0.0.0", "255.255.255.255"],
    ),
    TestCase(
        name="IPv6 — full form",
        input="Attacker used 2001:0db8:85a3:0000:0000:8a2e:0370:7334.",
        must_contain=["2001:0db8:85a3:0000:0000:8a2e:0370:7334"],
        must_not_split=["2001:0db8:85a3:0000:0000:8a2e:0370:7334"],
    ),
    TestCase(
        name="SHA-256 hash",
        input="Malware hash: 9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08",
        must_contain=["9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"],
        must_not_split=["9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"],
    ),
    TestCase(
        name="MD5 hash",
        input="File fingerprint: d41d8cd98f00b204e9800998ecf8427e",
        must_contain=["d41d8cd98f00b204e9800998ecf8427e"],
        must_not_split=["d41d8cd98f00b204e9800998ecf8427e"],
    ),
    TestCase(
        name="Windows file path",
        input="Payload dropped at C:\\Users\\Admin\\AppData\\Local\\malware.exe",
        must_contain=["C:\\Users\\Admin\\AppData\\Local\\malware.exe"],
        must_not_split=["C:\\Users\\Admin\\AppData\\Local\\malware.exe"],
    ),
    TestCase(
        name="Unix file path",
        input="Backdoor installed at /usr/local/bin/sshd_monitor",
        must_contain=["/usr/local/bin/sshd_monitor"],
        must_not_split=["/usr/local/bin/sshd_monitor"],
    ),
    TestCase(
        name="Hex string",
        input="Shellcode offset at 0xdeadbeef was identified.",
        must_contain=["0xdeadbeef"],
        must_not_split=["0xdeadbeef"],
    ),
    TestCase(
        name="Domain name",
        input="Beacon phoned home to evil-c2-server.onion every 60s.",
        must_contain=["evil-c2-server.onion"],
        must_not_split=["evil-c2-server.onion"],
    ),
    TestCase(
        name="Mixed IOCs in one sentence",
        input=(
            "Host 10.0.0.55 executed C:\\Windows\\System32\\cmd.exe "
            "with hash d41d8cd98f00b204e9800998ecf8427e "
            "matching CVE-2023-36884."
        ),
        must_contain=[
            "10.0.0.55",
            "C:\\Windows\\System32\\cmd.exe",
            "d41d8cd98f00b204e9800998ecf8427e",
            "CVE-2023-36884",
        ],
        must_not_split=[
            "10.0.0.55",
            "C:\\Windows\\System32\\cmd.exe",
            "d41d8cd98f00b204e9800998ecf8427e",
            "CVE-2023-36884",
        ],
    ),
    TestCase(
        name="Plain text — no IOCs",
        input="The adversary used spearphishing to gain initial access.",
        must_contain=["spearphishing", "adversary"],
        must_not_split=[],
    ),
]


def run_validation(verbose: bool = False) -> bool:
    print("=" * 60)
    print("  IOC Tokenizer - Validation Suite")
    print("=" * 60)

    passed = 0
    failed = 0

    for tc in TEST_CASES:
        tokens = pretokenize(tc.input)
        token_set = set(tokens)
        errors = []

        # Check must_contain
        for expected in tc.must_contain:
            if expected not in token_set:
                errors.append(f"  ✗ Missing atomic token: '{expected}'")

        # Check must_not_split — verify no IOC substring appears split
        for ioc in tc.must_not_split:
            # If the IOC is present as a whole token, it's fine
            if ioc in token_set:
                continue
            # Otherwise check if any part of it appears fragmented
            for tok in tokens:
                if tok in ioc and tok != ioc:
                    errors.append(f"  ✗ IOC was split — fragment found: '{tok}' from '{ioc}'")
                    break

        status = "PASS" if not errors else "FAIL"
        icon   = "✓" if not errors else "✗"
        print(f"\n[{status}] {icon} {tc.name}")

        if verbose or errors:
            print(f"  Input : {tc.input}")
            print(f"  Tokens: {tokens}")

        for err in errors:
            print(err)

        if errors:
            failed += 1
        else:
            passed += 1

    print("\n" + "=" * 60)
    print(f"  Results: {passed} passed, {failed} failed out of {len(TEST_CASES)} tests")
    print("=" * 60)

    return failed == 0


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SwiftSafe pre-tokenizer validation")
    parser.add_argument("--verbose", action="store_true", help="Show token splits for all tests")
    args = parser.parse_args()

    success = run_validation(verbose=args.verbose)
    sys.exit(0 if success else 1)
