"""Reading the hotel documentation: PDF pages and Markdown headings become sections."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Section:
    """One topic of the documentation (for example "Wi-Fi and connectivity")."""

    source: str  # file the section comes from
    title: str
    text: str

    @property
    def markdown(self) -> str:
        """The section as it is written into a prompt: a Markdown heading, then the text."""
        return f"## {self.title}\n\n{self.text}"


def load_sections(folder: str | Path) -> list[Section]:
    """Read every PDF and Markdown file of `folder` (sorted by file name) into sections.

    - **PDF**: one section per page. The first line of a page is its title and the last line
      is a footer (address, page number) that carries no information and is dropped.
    - **Markdown**: one section per `## ` heading.
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Documentation folder not found: {folder}")
    sections: list[Section] = []
    for path in sorted(folder.iterdir()):
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            sections += _read_pdf(path)
        elif suffix in (".md", ".markdown"):
            sections += _read_markdown(path)
    if not sections:
        raise ValueError(f"No PDF or Markdown documentation found in {folder}")
    return sections


def _read_pdf(path: Path) -> list[Section]:
    from pypdf import PdfReader

    sections = []
    for page in PdfReader(path).pages:
        lines = [line.strip() for line in (page.extract_text() or "").split("\n") if line.strip()]
        if len(lines) < 3:  # a title, some text and a footer at least
            continue
        sections.append(Section(path.name, lines[0], "\n".join(lines[1:-1])))
    return sections


def _read_markdown(path: Path) -> list[Section]:
    sections: list[Section] = []
    title, body = None, []

    def flush():
        text = "\n".join(body).strip()
        if title and text:
            sections.append(Section(path.name, title, text))

    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            flush()
            title, body = line[3:].strip(), []
        elif title is not None:
            body.append(line)
    flush()
    return sections
