from __future__ import annotations

import pandas as pd

from data_pipeline.utils import dedupe_tags, normalize_cve_mentions, normalize_text


def normalize_record(record: dict) -> dict:
    source = (record.get("source") or "").strip()
    text = normalize_text(record.get("text") or "")
    tags = dedupe_tags(list(record.get("domain_tags") or []))

    if source == "nvd_api_v2":
        text = normalize_cve_mentions(text)

    return {
        "source": source,
        "text": text,
        "domain_tags": tags,
    }


def clean_records(records: list[dict]) -> pd.DataFrame:
    normalized = [normalize_record(record) for record in records]
    frame = pd.DataFrame(normalized, columns=["source", "text", "domain_tags"])
    if frame.empty:
        return frame

    frame = frame[frame["text"].astype(bool)]
    frame = frame[frame["source"].astype(bool)]
    frame["domain_tags"] = frame["domain_tags"].apply(lambda tags: tags or ["untagged"])
    frame = frame.drop_duplicates(subset=["text"]).reset_index(drop=True)
    return frame
