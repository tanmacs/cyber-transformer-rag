from __future__ import annotations

from pathlib import Path
from typing import Any

from data_pipeline.utils import fetch_json, normalize_cve_id, normalize_text, severity_bucket


def _extract_english_description(cve: dict[str, Any]) -> str:
    descriptions = cve.get("descriptions", [])
    for item in descriptions:
        if item.get("lang") == "en":
            return normalize_text(item.get("value", ""))
    return ""


def _extract_base_score(cve: dict[str, Any]) -> float | None:
    metrics = cve.get("metrics", {})
    metric_paths = [
        ("cvssMetricV31", "cvssData"),
        ("cvssMetricV30", "cvssData"),
        ("cvssMetricV2", "cvssData"),
    ]
    for metric_key, cvss_key in metric_paths:
        entries = metrics.get(metric_key, [])
        if not entries:
            continue
        cvss_data = entries[0].get(cvss_key, {})
        base_score = cvss_data.get("baseScore")
        if base_score is not None:
            return float(base_score)
    return None


def _collect_cpe_matches(nodes: list[dict[str, Any]]) -> list[str]:
    matches: list[str] = []
    for node in nodes or []:
        for cpe in node.get("cpeMatch", []):
            criteria = cpe.get("criteria")
            if criteria:
                matches.append(criteria)
        children = node.get("children", [])
        if children:
            matches.extend(_collect_cpe_matches(children))
    return matches


def fetch_nvd_records(
    *,
    base_url: str,
    cache_dir: Path,
    use_cache: bool,
    api_key: str | None,
    results_per_page: int,
    delay_seconds: float,
    max_records: int | None,
    pub_start_date: str | None,
    pub_end_date: str | None,
) -> list[dict]:
    headers = {"apiKey": api_key} if api_key else {}
    cache_root = cache_dir / "nvd_api_v2"
    records: list[dict] = []
    start_index = 0

    while True:
        remaining = max_records - len(records) if max_records is not None else results_per_page
        if max_records is not None and remaining <= 0:
            break

        current_page_size = min(results_per_page, remaining) if max_records is not None else results_per_page
        params: dict[str, Any] = {
            "startIndex": start_index,
            "resultsPerPage": current_page_size,
            "noRejected": "",
        }
        if pub_start_date:
            params["pubStartDate"] = pub_start_date
        if pub_end_date:
            params["pubEndDate"] = pub_end_date

        payload = fetch_json(
            base_url,
            params=params,
            headers=headers,
            cache_path=cache_root / f"page_{start_index}_{current_page_size}.json",
            use_cache=use_cache,
            delay_seconds=delay_seconds,
        )

        vulnerabilities = payload.get("vulnerabilities", [])
        if not vulnerabilities:
            break

        for item in vulnerabilities:
            cve = item.get("cve", {})
            cve_id = normalize_cve_id(cve.get("id", ""))
            description = _extract_english_description(cve)
            if not cve_id or not description:
                continue

            base_score = _extract_base_score(cve)
            cpes: list[str] = []
            for configuration in cve.get("configurations", []):
                cpes.extend(_collect_cpe_matches(configuration.get("nodes", [])))

            text = f"{cve_id}: {description}"
            if base_score is not None:
                text += f" CVSS base score: {base_score}."
            if cpes:
                text += " Affected products: " + "; ".join(cpes[:10]) + "."

            records.append(
                {
                    "source": "nvd_api_v2",
                    "text": text,
                    "domain_tags": ["CVE", "network_security", severity_bucket(base_score)],
                }
            )

            if max_records is not None and len(records) >= max_records:
                return records

        total_results = payload.get("totalResults", 0)
        start_index += current_page_size
        if start_index >= total_results:
            break

    return records
