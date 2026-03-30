# Dataset Statistics

- Generated at (UTC): `2026-03-30T10:20:54.905685+00:00`
- Total records: `1705`
- Estimated tokens (`cl100k_base`): `346700`
- Average tokens per record: `203.34`

## Source Breakdown

- `nvd_api_v2`: `1000`
- `mitre_enterprise`: `705`

## Domain Distribution

- `cve`: `1000`
- `network_security`: `1000`
- `att_ck`: `705`
- `high_severity`: `495`
- `sub_technique`: `475`
- `windows`: `472`
- `macos`: `355`
- `linux`: `354`
- `medium_severity`: `278`
- `defense_evasion`: `216`
- `critical_severity`: `165`
- `persistence`: `127`
- `esxi`: `118`
- `privilege_escalation`: `110`
- `iaas`: `104`
- `network_devices`: `100`
- `pre`: `92`
- `office_suite`: `78`
- `credential_access`: `68`
- `saas`: `67`
- `discovery`: `50`
- `identity_provider`: `49`
- `resource_development`: `48`
- `containers`: `48`
- `execution`: `47`

## Run Metadata

- `nvd_max_records`: `1000`
- `nvd_results_per_page`: `500`
- `nvd_pub_start_date`: `2026-02-28T10:20:15.000Z`
- `nvd_pub_end_date`: `2026-03-30T10:20:15.000Z`
- `nvd_api_key_supplied`: `False`
- `internal_playbook_dir`: `C:\Users\tanma\OneDrive\Desktop\Tanmay\security-corpus\inputs\internal_playbooks`
- `cache_enabled`: `False`

## Gap Analysis

- No internal playbooks were present in `inputs/internal_playbooks/`, so the corpus is currently external-source heavy.
- NVD entries materially outweigh internal playbooks, which can bias the corpus toward vulnerability descriptions over response procedures.

## Assumptions

- The initial corpus build uses a 30-day NVD publication window by default to keep the first run fast and reproducible without an API key.
- Internal playbooks are ingested from local files in inputs/internal_playbooks and were omitted if the folder is empty.
- MITRE ATT&CK ingestion includes enterprise tactics and attack patterns, excluding revoked and deprecated objects.
