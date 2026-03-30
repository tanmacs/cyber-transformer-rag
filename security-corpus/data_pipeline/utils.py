from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any

import requests
from bs4 import BeautifulSoup


LOGGER = logging.getLogger("security_corpus")
CVE_ID_PATTERN = re.compile(r"\bCVE[-_ ]?(\d{4})[-_ ]?(\d{4,})\b", re.IGNORECASE)
HTML_PATTERN = re.compile(r"</?[a-zA-Z][^>]*>|&#?\w+;")


def configure_logging(verbose: bool = True) -> None:
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def strip_html(text: str) -> str:
    if not text:
        return ""
    if not HTML_PATTERN.search(text):
        return text
    return BeautifulSoup(text, "html.parser").get_text(" ", strip=True)


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def force_utf8(text: str) -> str:
    return (text or "").encode("utf-8", "ignore").decode("utf-8")


def normalize_text(text: str) -> str:
    text = strip_html(text)
    text = force_utf8(text)
    return normalize_whitespace(text)


def normalize_tag(value: str) -> str:
    cleaned = normalize_whitespace(value).lower()
    cleaned = re.sub(r"[^a-z0-9]+", "_", cleaned).strip("_")
    return cleaned


def normalize_cve_id(value: str) -> str:
    match = CVE_ID_PATTERN.search(value or "")
    if not match:
        return normalize_whitespace(value)
    year, identifier = match.groups()
    return f"CVE-{year}-{identifier}"


def normalize_cve_mentions(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        year, identifier = match.groups()
        return f"CVE-{year}-{identifier}"

    return CVE_ID_PATTERN.sub(replace, text or "")


def dedupe_tags(values: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        tag = normalize_tag(value)
        if not tag or tag in seen:
            continue
        seen.add(tag)
        deduped.append(tag)
    return deduped


def save_json(payload: Any, destination: Path) -> None:
    ensure_directory(destination.parent)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_json(source: Path) -> Any:
    return json.loads(source.read_text(encoding="utf-8"))


def fetch_json(
    url: str,
    *,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: int = 60,
    cache_path: Path | None = None,
    use_cache: bool = False,
    delay_seconds: float = 0.0,
) -> dict[str, Any]:
    if use_cache and cache_path and cache_path.exists():
        LOGGER.info("Loading cached response from %s", cache_path)
        return load_json(cache_path)

    LOGGER.info("Fetching %s", url)
    response = requests.get(url, params=params, headers=headers, timeout=timeout)
    response.raise_for_status()
    payload = response.json()

    if cache_path:
        save_json(payload, cache_path)

    if delay_seconds > 0:
        time.sleep(delay_seconds)

    return payload


def severity_bucket(base_score: float | None) -> str:
    if base_score is None:
        return "unknown_severity"
    if base_score >= 9.0:
        return "critical_severity"
    if base_score >= 7.0:
        return "high_severity"
    if base_score >= 4.0:
        return "medium_severity"
    if base_score > 0:
        return "low_severity"
    return "none_severity"
