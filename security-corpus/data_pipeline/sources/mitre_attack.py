from __future__ import annotations

from pathlib import Path

from data_pipeline.utils import fetch_json, normalize_text


MITRE_CACHE_NAME = "mitre_enterprise_attack.json"


def _attack_id(external_references: list[dict]) -> str | None:
    for reference in external_references or []:
        external_id = reference.get("external_id")
        if external_id and external_id.startswith("T"):
            return external_id
    return None


def fetch_mitre_records(
    *,
    mitre_url: str,
    cache_dir: Path,
    use_cache: bool,
) -> list[dict]:
    payload = fetch_json(
        mitre_url,
        cache_path=cache_dir / MITRE_CACHE_NAME,
        use_cache=use_cache,
    )

    records: list[dict] = []
    for obj in payload.get("objects", []):
        if obj.get("revoked") or obj.get("x_mitre_deprecated"):
            continue

        obj_type = obj.get("type")
        description = normalize_text(obj.get("description", ""))
        if not description:
            continue

        if obj_type == "attack-pattern":
            attack_id = _attack_id(obj.get("external_references", []))
            if not attack_id:
                continue
            name = normalize_text(obj.get("name", ""))
            phases = [
                phase.get("phase_name", "")
                for phase in obj.get("kill_chain_phases", [])
                if phase.get("phase_name")
            ]
            platforms = obj.get("x_mitre_platforms", [])
            text = f"{attack_id}: {name}. {description}"
            if platforms:
                text += f" Platforms: {', '.join(platforms)}."
            tags = ["ATT&CK"] + phases + platforms
            if obj.get("x_mitre_is_subtechnique"):
                tags.append("sub-technique")
            records.append(
                {
                    "source": "mitre_enterprise",
                    "text": text,
                    "domain_tags": tags,
                }
            )

        if obj_type == "x-mitre-tactic":
            name = normalize_text(obj.get("name", ""))
            shortname = obj.get("x_mitre_shortname", "")
            text = f"{name}: {description}"
            records.append(
                {
                    "source": "mitre_enterprise",
                    "text": text,
                    "domain_tags": ["ATT&CK", shortname or name, "tactic"],
                }
            )

    return records

