"""Local, small knowledge-source loader for grounded responses."""

from __future__ import annotations

import re
from pathlib import Path
from typing import NamedTuple


KNOWLEDGE_DIR = Path(__file__).with_name("knowledge_base")


class SourceExcerpt(NamedTuple):
    source: str
    text: str


def load_sources(directory: str | Path = KNOWLEDGE_DIR) -> list[SourceExcerpt]:
    excerpts: list[SourceExcerpt] = []
    for path in sorted(Path(directory).glob("*.txt")):
        content = path.read_text(encoding="utf-8").strip()
        if content:
            excerpts.append(SourceExcerpt(path.stem, content[:8000]))
    return excerpts


def retrieve(query: str, limit: int = 3, directory: str | Path = KNOWLEDGE_DIR) -> list[SourceExcerpt]:
    """Use simple local term matching; never imply that this is a clinical guideline engine."""
    terms = {term.lower() for term in re.findall(r"[a-zA-Z]{3,}", query)}
    scored: list[tuple[int, SourceExcerpt]] = []
    for source in load_sources(directory):
        words = set(re.findall(r"[a-zA-Z]{3,}", source.text.lower()))
        score = len(terms & words)
        if score:
            scored.append((score, source))
    scored.sort(key=lambda item: (-item[0], item[1].source))
    return [excerpt for _, excerpt in scored[:max(0, limit)]]
