from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from data_pipeline.utils import ensure_directory


def export_jsonl(frame: pd.DataFrame, destination: Path) -> None:
    ensure_directory(destination.parent)
    with destination.open("w", encoding="utf-8") as handle:
        for row in frame.to_dict(orient="records"):
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

