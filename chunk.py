"""Page chunking: overlapping fixed-size word windows.

Title + first sentence prepended to every chunk for entity resolution.
Empirically: avg page ≈ 2217 words, MiniLM limit ≈ 180 tokens.
Chunking ensures deep-page content is reachable by the cross-encoder reranker.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

# Chunking parameters (word-level)
CHUNK_SIZE = 150   # words per chunk
OVERLAP = 50       # overlapping words between consecutive chunks
STEP = CHUNK_SIZE - OVERLAP  # stride between chunk start positions


@dataclass
class Chunk:
    page_id: int    # corpus page this chunk belongs to
    chunk_id: int   # position index within the page (0 = first chunk)
    text: str       # prefix + chunk window text passed to the encoder


def _first_sentence(content: str) -> str:
    """Extract the first sentence from page content (up to 300 chars)."""
    for sep in (".", "!", "?"):
        idx = content.find(sep)
        if idx != -1 and idx < 300:
            return content[: idx + 1].strip()
    return content[:200].strip()


def chunk_entry(record: Dict[str, Any]) -> List[Chunk]:
    """Split a single page into overlapping 150-word chunks.

    Each chunk is prefixed with the page title + first sentence to preserve
    entity context across all windows (entity resolution anchor).
    Pages shorter than CHUNK_SIZE are returned as a single chunk.
    """
    page_id = int(record["page_id"])
    title = str(record.get("title", "")).strip()
    content = str(record.get("content", "")).strip()
    first_sent = _first_sentence(content)
    prefix = f"{title}. {first_sent}"

    words = content.split()

    if len(words) <= CHUNK_SIZE:
        return [Chunk(page_id=page_id, chunk_id=0,
                      text=f"{prefix}\n\n{content}".strip())]

    chunks = []
    chunk_id = 0
    start = 0
    while start < len(words):
        window = " ".join(words[start: start + CHUNK_SIZE])
        text = f"{prefix}\n\n{window}"
        chunks.append(Chunk(page_id=page_id, chunk_id=chunk_id, text=text))
        chunk_id += 1
        start += STEP

    return chunks


def chunk_corpus(records: List[Dict[str, Any]]) -> List[Chunk]:
    """Chunk all corpus pages and return a flat list of Chunk objects."""
    chunks: List[Chunk] = []
    for record in records:
        chunks.extend(chunk_entry(record))
    return chunks