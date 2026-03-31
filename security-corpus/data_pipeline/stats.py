from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import tiktoken

from data_pipeline.utils import ensure_directory


def estimate_token_count(texts: list[str]) -> int:
    encoder = tiktoken.get_encoding("cl100k_base")
    return sum(len(encoder.encode(text)) for text in texts)


def build_stats_summary(
    frame: pd.DataFrame,
    *,
    assumptions: list[str],
    run_metadata: dict[str, Any],
) -> dict[str, Any]:
    source_breakdown = frame["source"].value_counts().to_dict() if not frame.empty else {}
    tag_counter: Counter[str] = Counter()
    for tags in frame.get("domain_tags", []):
        tag_counter.update(tags)

    texts = frame["text"].tolist() if not frame.empty else []
    token_count = estimate_token_count(texts) if texts else 0

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_records": int(len(frame)),
        "source_breakdown": source_breakdown,
        "domain_distribution": dict(tag_counter.most_common()),
        "estimated_tokens_cl100k_base": token_count,
        "average_tokens_per_record": round(token_count / len(frame), 2) if len(frame) else 0,
        "assumptions": assumptions,
        "run_metadata": run_metadata,
    }


def render_stats_markdown(summary: dict[str, Any]) -> str:
    source_breakdown = summary["source_breakdown"]
    domain_distribution = summary["domain_distribution"]
    assumptions = summary["assumptions"]
    run_metadata = summary["run_metadata"]

    lines = [
        "# Dataset Statistics",
        "",
        f"- Generated at (UTC): `{summary['generated_at_utc']}`",
        f"- Total records: `{summary['total_records']}`",
        f"- Estimated tokens (`cl100k_base`): `{summary['estimated_tokens_cl100k_base']}`",
        f"- Average tokens per record: `{summary['average_tokens_per_record']}`",
        "",
        "## Source Breakdown",
        "",
    ]

    if source_breakdown:
        for source, count in source_breakdown.items():
            lines.append(f"- `{source}`: `{count}`")
    else:
        lines.append("- No records were generated.")

    lines.extend(["", "## Domain Distribution", ""])
    if domain_distribution:
        for tag, count in list(domain_distribution.items())[:25]:
            lines.append(f"- `{tag}`: `{count}`")
    else:
        lines.append("- No domain tags were generated.")

    lines.extend(["", "## Run Metadata", ""])
    for key, value in run_metadata.items():
        lines.append(f"- `{key}`: `{value}`")

    lines.extend(["", "## Gap Analysis", ""])
    playbook_count = source_breakdown.get("internal_playbook", 0)
    nvd_count = source_breakdown.get("nvd_api_v2", 0)
    mitre_count = source_breakdown.get("mitre_enterprise", 0)

    if playbook_count == 0:
        lines.append("- No internal playbooks were present in `inputs/internal_playbooks/`, so the corpus is currently external-source heavy.")
    if nvd_count > max(playbook_count, 1) * 10:
        lines.append("- NVD entries materially outweigh internal playbooks, which can bias the corpus toward vulnerability descriptions over response procedures.")
    if mitre_count and playbook_count and mitre_count > playbook_count * 5:
        lines.append("- MITRE ATT&CK knowledge outweighs internal operating guidance, so defensive workflow language may still be underrepresented.")
    if not any(line.startswith("- ") and "bias" in line for line in lines[-3:]):
        lines.append("- The current source mix is reasonably balanced for the available inputs, but adding more internal content would still improve task specificity.")

    lines.extend(["", "## Assumptions", ""])
    for item in assumptions:
        lines.append(f"- {item}")

    lines.append("")
    return "\n".join(lines)


def write_stats_report(summary: dict[str, Any], destination: Path) -> None:
    ensure_directory(destination.parent)
    destination.write_text(render_stats_markdown(summary), encoding="utf-8")

