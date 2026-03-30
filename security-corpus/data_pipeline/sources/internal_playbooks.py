from __future__ import annotations

from pathlib import Path

from bs4 import BeautifulSoup
from docx import Document
from markdown import markdown
from pypdf import PdfReader

from data_pipeline.utils import normalize_text


SUPPORTED_SUFFIXES = {".md", ".markdown", ".html", ".htm", ".txt", ".docx", ".pdf"}


def _read_markdown(path: Path) -> str:
    html = markdown(path.read_text(encoding="utf-8"))
    return BeautifulSoup(html, "html.parser").get_text(" ", strip=True)


def _read_html(path: Path) -> str:
    return BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser").get_text(" ", strip=True)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _read_docx(path: Path) -> str:
    document = Document(path)
    return "\n".join(paragraph.text for paragraph in document.paragraphs)


def _read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def _load_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".md", ".markdown"}:
        return _read_markdown(path)
    if suffix in {".html", ".htm"}:
        return _read_html(path)
    if suffix == ".txt":
        return _read_text(path)
    if suffix == ".docx":
        return _read_docx(path)
    if suffix == ".pdf":
        return _read_pdf(path)
    return ""


def fetch_internal_playbook_records(playbooks_dir: Path) -> list[dict]:
    if not playbooks_dir.exists():
        return []

    records: list[dict] = []
    for path in sorted(playbooks_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue

        content = normalize_text(_load_file(path))
        if not content:
            continue

        title = path.stem.replace("_", " ").replace("-", " ")
        text = f"{title}: {content}"
        records.append(
            {
                "source": "internal_playbook",
                "text": text,
                "domain_tags": ["internal_playbook", "soc", path.stem],
            }
        )

    return records

