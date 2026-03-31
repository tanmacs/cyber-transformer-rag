# Security Corpus Pipeline

This project builds a high-signal, domain-specific security corpus from three modular sources:

- MITRE ATT&CK enterprise STIX data
- NVD REST API v2.0 CVE records
- Internal security playbooks stored locally

The pipeline cleans and normalizes the records, exports a strict JSONL corpus, and generates a dataset statistics report for downstream tokenizer and MLOps handoff.

## Project Layout

```text
security-corpus/
├── cache/
├── data_pipeline/
│   ├── build_corpus.py
│   ├── cleaning.py
│   ├── config.py
│   ├── exporters.py
│   ├── stats.py
│   ├── utils.py
│   └── sources/
│       ├── internal_playbooks.py
│       ├── mitre_attack.py
│       └── nvd_api.py
├── inputs/
│   └── internal_playbooks/
├── outputs/
├── README.md
└── requirements.txt
```

## What Gets Extracted

### MITRE ATT&CK

- Enterprise attack patterns and tactics
- `name`
- `description`
- `x_mitre_platforms`
- Kill chain phase names for tagging

### NVD API v2.0

- English CVE descriptions
- CVSS base score from v3.1, v3.0, or v2 metrics
- CPE match criteria for affected products
- Pagination with `startIndex` and `resultsPerPage`
- Delay-aware requests for safer rate-limit handling

### Internal Security Playbooks

- Local `.md`, `.markdown`, `.html`, `.htm`, `.txt`, `.docx`, and `.pdf` files
- HTML is stripped with BeautifulSoup
- Markdown is converted to HTML first, then reduced to clean text

## Cleaning and Normalization

- HTML stripping with BeautifulSoup
- UTF-8 enforcement
- whitespace normalization
- CVE identifier normalization to `CVE-YYYY-NNNNN`
- duplicate removal by `text`

## Setup

```powershell
cd security-corpus
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the Pipeline

The default run is intentionally practical for a first build:

- MITRE ATT&CK full enterprise ingestion
- NVD limited to the most recent 30-day publication window
- NVD capped at 1,000 records unless you override it
- local internal playbook ingestion from `inputs/internal_playbooks/`

```powershell
python -m data_pipeline.build_corpus
```

### Recommended production-scale run

Set an NVD API key first so you can safely scale the NVD pull:

```powershell
$env:NVD_API_KEY="your_api_key_here"
python -m data_pipeline.build_corpus --nvd-api-key $env:NVD_API_KEY --nvd-max-records 5000 --nvd-results-per-page 1000
```

### Disable cache when needed

```powershell
python -m data_pipeline.build_corpus --no-cache
```

## Output Artifacts

After a successful run, the pipeline writes:

- `corpus_v1.jsonl`
- `dataset_stats.md`
- `outputs/run_summary.json`

## Internal Playbook Handoff

Drop internal documents into:

```text
inputs/internal_playbooks/
```

Supported formats:

- `.md`
- `.markdown`
- `.html`
- `.htm`
- `.txt`
- `.docx`
- `.pdf`

## Example JSONL Schema

```json
{"source": "nvd_api_v2", "text": "CVE-2026-0001: Example vulnerability description. CVSS base score: 8.8. Affected products: cpe:2.3:a:vendor:product:1.0:*:*:*:*:*:*:*.", "domain_tags": ["cve", "network_security", "high_severity"]}
{"source": "mitre_enterprise", "text": "T1566.001: Spearphishing Attachment. Adversaries may send emails ...", "domain_tags": ["att_ck", "initial_access", "windows"]}
```

## Notes for Tanmay and Vibin

- Tanmay (Tokenizer): consume `corpus_v1.jsonl`
- Vibin (MLOps): review `dataset_stats.md` and `outputs/run_summary.json`

The first run will be externally weighted unless internal playbooks are added.
