"""Offline index build – run once locally via scripts/build_index.py.

Artifacts produced under artifacts/:
  faiss_pages.index   — one L2-normalised vector per page (stage-1 dense retrieval)
  page_ids.npy        — page_id for each row in faiss_pages
  chunk_vectors.npy   — chunk-level embeddings (used by cross-encoder reranker)
  chunk_page_ids.npy  — page_id for each chunk row
  chunk_ids.npy       — chunk position index (0 = first chunk of a page)
  bm25.pkl            — BM25 inverted index over full page text
  index_meta.json     — build metadata (num pages, num chunks, model name)
"""
from __future__ import annotations

import json
import math
import pickle
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np

from chunk import chunk_corpus
from embed import embed_texts
from utils import ARTIFACTS_DIR, ensure_artifacts_dir, iter_entries

# --- Artifact file names ---
FAISS_PAGES_NAME = "faiss_pages.index"
PAGE_IDS_NAME = "page_ids.npy"
CHUNK_VECTORS_NAME = "chunk_vectors.npy"
CHUNK_PAGE_IDS_NAME = "chunk_page_ids.npy"
CHUNK_IDS_NAME = "chunk_ids.npy"
BM25_NAME = "bm25.pkl"
INDEX_META_NAME = "index_meta.json"


def tokenize(text: str) -> List[str]:
    """Lowercase alphanumeric tokeniser shared by BM25 build and retrieval."""
    return re.findall(r"[a-z0-9]+", text.lower())


def build_bm25(records: List[Dict[str, Any]], out_dir: Path) -> None:
    """Build and save a BM25 inverted index over full page text.
    
    Title tokens are repeated 3x to boost title-match relevance.
    """
    page_ids: List[int] = []
    docs: List[List[str]] = []
    for r in records:
        page_ids.append(int(r["page_id"]))
        text = (str(r.get("title", "")) + " ") * 3 + str(r.get("content", ""))
        docs.append(tokenize(text))

    n = len(docs)
    doc_len = np.array([len(d) for d in docs], dtype=np.float32)
    avgdl = float(doc_len.mean()) if n else 1.0

    # Build inverted index: vocab → postings list of (doc_idx, term_freq)
    vocab: Dict[str, int] = {}
    postings: List[List[Tuple[int, int]]] = []
    df: List[int] = []
    for i, toks in enumerate(docs):
        for term, count in Counter(toks).items():
            tid = vocab.setdefault(term, len(vocab))
            if tid == len(postings):
                postings.append([])
                df.append(0)
            postings[tid].append((i, count))
            df[tid] += 1

    # IDF with BM25 smoothing
    idf = np.array(
        [math.log(1 + (n - f + 0.5) / (f + 0.5)) for f in df],
        dtype=np.float32,
    )

    # Flatten postings into CSR-style arrays for fast retrieval
    indptr = np.zeros(len(postings) + 1, dtype=np.int64)
    for tid, pl in enumerate(postings):
        indptr[tid + 1] = indptr[tid] + len(pl)
    doc_idx = np.empty(indptr[-1], dtype=np.int32)
    tfs = np.empty(indptr[-1], dtype=np.float32)
    for tid, pl in enumerate(postings):
        s = indptr[tid]
        for j, (d, c) in enumerate(pl):
            doc_idx[s + j] = d
            tfs[s + j] = c

    with open(out_dir / BM25_NAME, "wb") as f:
        pickle.dump({
            "vocab": vocab, "indptr": indptr, "doc_idx": doc_idx,
            "tfs": tfs, "idf": idf, "doc_len": doc_len,
            "avgdl": avgdl, "page_ids": page_ids,
        }, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"BM25 saved: {n} docs, vocab={len(vocab)}")


def build_index(
    *,
    entries_dir: Optional[Path] = None,
    artifacts_dir: Optional[Path] = None,
) -> None:
    """Build all artifacts offline. Called locally; not run by the autograder."""
    out_dir = artifacts_dir or ensure_artifacts_dir()
    records = list(iter_entries(entries_dir))

    # --- Stage 1: page-level FAISS index (title + first 180 words) ---
    print("Embedding pages (stage-1)...")
    page_texts = []
    page_ids = []
    for r in records:
        page_ids.append(int(r["page_id"]))
        title = str(r.get("title", "")).strip()
        content = str(r.get("content", "")).strip()
        words = content.split()[:180]
        page_texts.append(f"{title}\n\n{' '.join(words)}")

    page_vectors = embed_texts(page_texts, batch_size=128)
    np.save(out_dir / PAGE_IDS_NAME, np.array(page_ids, dtype=np.int64))

    faiss_pages = faiss.IndexFlatIP(page_vectors.shape[1])
    faiss_pages.add(page_vectors)
    faiss.write_index(faiss_pages, str(out_dir / FAISS_PAGES_NAME))
    print(f"Page FAISS saved: {faiss_pages.ntotal} vectors.")

    # --- Stage 2: chunk-level embeddings (used by cross-encoder reranker) ---
    print("Chunking corpus...")
    chunks = chunk_corpus(records)
    print(f"Total chunks: {len(chunks)}")

    print("Embedding chunks...")
    chunk_vectors = embed_texts([c.text for c in chunks], batch_size=128)
    chunk_page_ids = np.array([c.page_id for c in chunks], dtype=np.int64)
    chunk_ids = np.array([c.chunk_id for c in chunks], dtype=np.int32)

    np.save(out_dir / CHUNK_VECTORS_NAME, chunk_vectors)
    np.save(out_dir / CHUNK_PAGE_IDS_NAME, chunk_page_ids)
    np.save(out_dir / CHUNK_IDS_NAME, chunk_ids)
    print("Chunk vectors saved.")

    # --- BM25 over full page text ---
    print("Building BM25...")
    build_bm25(records, out_dir)

    meta = {
        "num_pages": len(records),
        "num_chunks": len(chunks),
        "model": "sentence-transformers/all-MiniLM-L6-v2",
    }
    (out_dir / INDEX_META_NAME).write_text(json.dumps(meta), encoding="utf-8")
    print("Build complete.")


# --- Artifact loaders (called at query time by retrieve.py) ---

def load_page_faiss(artifacts_dir: Optional[Path] = None) -> faiss.Index:
    """Load the page-level FAISS index from disk."""
    root = artifacts_dir or ARTIFACTS_DIR
    return faiss.read_index(str(root / FAISS_PAGES_NAME))


def load_page_ids(artifacts_dir: Optional[Path] = None) -> np.ndarray:
    """Load the page_id array corresponding to FAISS index rows."""
    return np.load((artifacts_dir or ARTIFACTS_DIR) / PAGE_IDS_NAME)


def load_chunk_data(
    artifacts_dir: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load chunk vectors, their page_ids, and chunk position indices."""
    root = artifacts_dir or ARTIFACTS_DIR
    vectors = np.load(root / CHUNK_VECTORS_NAME)
    page_ids = np.load(root / CHUNK_PAGE_IDS_NAME)
    chunk_ids = np.load(root / CHUNK_IDS_NAME)
    return vectors, page_ids, chunk_ids


def load_bm25(artifacts_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Load the BM25 index from disk."""
    root = artifacts_dir or ARTIFACTS_DIR
    with open(root / BM25_NAME, "rb") as f:
        return pickle.load(f)


def load_metadata(artifacts_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Load build metadata from index_meta.json."""
    root = artifacts_dir or ARTIFACTS_DIR
    return json.loads((root / INDEX_META_NAME).read_text(encoding="utf-8"))