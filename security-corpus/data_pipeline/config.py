from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class PipelineConfig:
    root_dir: Path
    cache_dir: Path
    outputs_dir: Path
    playbooks_dir: Path
    mitre_url: str
    nvd_base_url: str
    nvd_api_key: str | None
    nvd_results_per_page: int
    nvd_delay_seconds: float
    nvd_max_records: int | None
    nvd_pub_start_date: str | None
    nvd_pub_end_date: str | None
    use_cache: bool

