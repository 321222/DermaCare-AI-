"""Optional local lexical retrieval wrapper used by Dr. Derma."""

from __future__ import annotations

from knowledge import SourceExcerpt, retrieve


def get_relevant_context(query: str, limit: int = 3) -> list[SourceExcerpt]:
    """Fetch relevant excerpts from the bundled local reference files."""
    return retrieve(query, limit=limit)
