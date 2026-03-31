from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from data_pipeline.cleaning import clean_records
from data_pipeline.config import PipelineConfig
from data_pipeline.exporters import export_jsonl
from data_pipeline.sources.internal_playbooks import fetch_internal_playbook_records
from data_pipeline.sources.mitre_attack import fetch_mitre_records
from data_pipeline.sources.nvd_api import fetch_nvd_records
from data_pipeline.stats import build_stats_summary, write_stats_report
from data_pipeline.utils import LOGGER, configure_logging, ensure_directory


def default_recent_window() -> tuple[str, str]:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=30)
    fmt = "%Y-%m-%dT%H:%M:%S.000Z"
    return start.strftime(fmt), end.strftime(fmt)


def parse_args() -> argparse.Namespace:
    root_dir = Path(__file__).resolve().parents[1]
    default_start, default_end = default_recent_window()

    parser = argparse.ArgumentParser(description="Build a domain-specific security corpus.")
    parser.add_argument("--root-dir", type=Path, default=root_dir)
    parser.add_argument(
        "--mitre-url",
        default="https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json",
    )
    parser.add_argument(
        "--nvd-base-url",
        default="https://services.nvd.nist.gov/rest/json/cves/2.0",
    )
    parser.add_argument("--nvd-api-key", default=None)
    parser.add_argument("--nvd-results-per-page", type=int, default=500)
    parser.add_argument("--nvd-delay-seconds", type=float, default=None)
    parser.add_argument("--nvd-max-records", type=int, default=1000)
    parser.add_argument("--nvd-pub-start-date", default=default_start)
    parser.add_argument("--nvd-pub-end-date", default=default_end)
    parser.add_argument("--no-cache", action="store_true")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> PipelineConfig:
    delay_seconds = args.nvd_delay_seconds
    if delay_seconds is None:
        delay_seconds = 0.8 if args.nvd_api_key else 6.2

    return PipelineConfig(
        root_dir=args.root_dir,
        cache_dir=args.root_dir / "cache",
        outputs_dir=args.root_dir / "outputs",
        playbooks_dir=args.root_dir / "inputs" / "internal_playbooks",
        mitre_url=args.mitre_url,
        nvd_base_url=args.nvd_base_url,
        nvd_api_key=args.nvd_api_key,
        nvd_results_per_page=args.nvd_results_per_page,
        nvd_delay_seconds=delay_seconds,
        nvd_max_records=args.nvd_max_records,
        nvd_pub_start_date=args.nvd_pub_start_date,
        nvd_pub_end_date=args.nvd_pub_end_date,
        use_cache=not args.no_cache,
    )


def main() -> int:
    configure_logging()
    args = parse_args()
    config = build_config(args)

    ensure_directory(config.cache_dir)
    ensure_directory(config.outputs_dir)
    ensure_directory(config.playbooks_dir)

    assumptions = [
        "The initial corpus build uses a 30-day NVD publication window by default to keep the first run fast and reproducible without an API key.",
        "Internal playbooks are ingested from local files in inputs/internal_playbooks and were omitted if the folder is empty.",
        "MITRE ATT&CK ingestion includes enterprise tactics and attack patterns, excluding revoked and deprecated objects.",
    ]

    LOGGER.info("Fetching MITRE ATT&CK records")
    mitre_records = fetch_mitre_records(
        mitre_url=config.mitre_url,
        cache_dir=config.cache_dir,
        use_cache=config.use_cache,
    )

    LOGGER.info("Fetching NVD records")
    nvd_records = fetch_nvd_records(
        base_url=config.nvd_base_url,
        cache_dir=config.cache_dir,
        use_cache=config.use_cache,
        api_key=config.nvd_api_key,
        results_per_page=config.nvd_results_per_page,
        delay_seconds=config.nvd_delay_seconds,
        max_records=config.nvd_max_records,
        pub_start_date=config.nvd_pub_start_date,
        pub_end_date=config.nvd_pub_end_date,
    )

    LOGGER.info("Fetching internal playbook records")
    playbook_records = fetch_internal_playbook_records(config.playbooks_dir)

    all_records = mitre_records + nvd_records + playbook_records
    frame = clean_records(all_records)

    corpus_path = config.root_dir / "corpus_v1.jsonl"
    stats_path = config.root_dir / "dataset_stats.md"
    run_summary_path = config.outputs_dir / "run_summary.json"

    export_jsonl(frame, corpus_path)
    summary = build_stats_summary(
        frame,
        assumptions=assumptions,
        run_metadata={
            "nvd_max_records": config.nvd_max_records,
            "nvd_results_per_page": config.nvd_results_per_page,
            "nvd_pub_start_date": config.nvd_pub_start_date,
            "nvd_pub_end_date": config.nvd_pub_end_date,
            "nvd_api_key_supplied": bool(config.nvd_api_key),
            "internal_playbook_dir": str(config.playbooks_dir),
            "cache_enabled": config.use_cache,
        },
    )
    write_stats_report(summary, stats_path)
    run_summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    LOGGER.info("Corpus written to %s", corpus_path)
    LOGGER.info("Stats report written to %s", stats_path)
    LOGGER.info("Total cleaned records: %s", len(frame))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
