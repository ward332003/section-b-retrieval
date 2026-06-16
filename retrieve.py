"""Hybrid retrieval: page-level dense + BM25 + cross-encoder rerank."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from sentence_transformers import CrossEncoder

from embed import embed_queries
from index import load_bm25, load_page_faiss, load_page_ids, tokenize
from utils import ENTRIES_DIR, K_EVAL

# --- Retrieval hyperparameters ---
POOL_N = 100          # number of candidates retrieved from each source (dense + BM25)
DENSE_WEIGHT = 2.0    # RRF weight for dense retrieval scores
BM25_WEIGHT = 1.0     # RRF weight for BM25 scores
RRF_K = 60            # RRF smoothing constant
RERANK_TOP = 50       # top-N candidates passed to the cross-encoder for reranking
BM25_K1 = 1.5         # BM25 term-frequency saturation parameter
BM25_B = 0.75         # BM25 document-length normalisation parameter

# Cross-encoder fine-tuned on synthetic corpus data for reranking
CROSS_ENCODER_MODEL = "artifacts/cross_encoder_synthetic"

# Module-level cache so resources are loaded only once per process
_page_faiss = None
_page_ids: np.ndarray | None = None
_bm25: Dict[str, Any] | None = None
_cross_encoder: CrossEncoder | None = None
_page_texts: Dict[int, str] | None = None


def _load_resources(artifacts_dir: Optional[Path] = None) -> None:
    """Load all retrieval resources into module-level cache (runs once)."""
    global _page_faiss, _page_ids, _bm25, _cross_encoder, _page_texts
    if _page_faiss is not None:
        return

    _page_faiss = load_page_faiss(artifacts_dir)
    _page_ids = load_page_ids(artifacts_dir)
    _bm25 = load_bm25(artifacts_dir)
    _cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
    _cross_encoder.predict([("warmup", "warmup")])  # warm up GPU kernels

    # Load truncated page texts (title + first 300 words) for cross-encoder input
    _page_texts = {}
    for path in sorted(ENTRIES_DIR.glob("*.json")):
        d = json.loads(path.read_text(encoding="utf-8"))
        pid = int(d.get("page_id", path.stem))
        title = str(d.get("title", "")).strip()
        content = str(d.get("content", "")).strip()
        words = content.split()[:300]
        _page_texts[pid] = f"{title}\n\n{' '.join(words)}"


def _bm25_scores(query: str) -> np.ndarray:
    """Compute BM25 scores for all corpus pages given a query string."""
    bm = _bm25
    scores = np.zeros(len(bm["doc_len"]), dtype=np.float32)
    norm = BM25_K1 * (1 - BM25_B + BM25_B * bm["doc_len"] / bm["avgdl"])
    for term in tokenize(query):
        tid = bm["vocab"].get(term)
        if tid is None:
            continue
        s, e = bm["indptr"][tid], bm["indptr"][tid + 1]
        d, tf = bm["doc_idx"][s:e], bm["tfs"][s:e]
        scores[d] += bm["idf"][tid] * tf * (BM25_K1 + 1) / (tf + norm[d])
    return scores


def search_batch(
    queries: List[str],
    *,
    top_k: int = K_EVAL,
    artifacts_dir: Optional[Path] = None,
    pool_n: int = POOL_N,
    dense_weight: float = DENSE_WEIGHT,
    bm25_weight: float = BM25_WEIGHT,
    rerank_top: int = RERANK_TOP,
) -> List[List[int]]:
    """Retrieve and rerank pages for a batch of queries.

    Pipeline:
      1. Dense retrieval via FAISS (all-MiniLM-L6-v2 embeddings).
      2. BM25 lexical retrieval.
      3. Reciprocal Rank Fusion (RRF) to merge both candidate lists.
      4. Cross-encoder reranking of the top candidates.

    Returns a list of ranked page_id lists (most relevant first).
    """
    _load_resources(artifacts_dir)

    query_vectors = embed_queries(queries)
    if query_vectors.size == 0:
        return [[] for _ in queries]

    # Dense retrieval: fetch top-pool_n pages per query from FAISS
    k = min(pool_n, _page_faiss.ntotal)
    _, dense_idx = _page_faiss.search(query_vectors, k)

    ranked: List[List[int]] = []
    for qi, query in enumerate(queries):
        fused: Dict[int, float] = {}

        # Add dense RRF scores
        for rank, idx in enumerate(dense_idx[qi]):
            if idx < 0:
                continue
            pid = int(_page_ids[int(idx)])
            fused[pid] = fused.get(pid, 0.0) + dense_weight / (RRF_K + rank)

        # Add BM25 RRF scores
        bm_scores = _bm25_scores(query)
        bm_top = np.argsort(-bm_scores)[:pool_n]
        for rank, di in enumerate(bm_top):
            pid = int(_bm25["page_ids"][int(di)])
            fused[pid] = fused.get(pid, 0.0) + bm25_weight / (RRF_K + rank)

        # Select top candidates by fused RRF score for reranking
        sorted_pids = [
            pid for pid, _ in
            sorted(fused.items(), key=lambda x: -x[1])[:rerank_top]
        ]

        # Cross-encoder reranking: score (query, page_text) pairs
        pairs = [(query, _page_texts.get(pid, "")) for pid in sorted_pids]
        ce_scores = _cross_encoder.predict(pairs)
        ce_order = np.argsort(-ce_scores)[:top_k]
        ranked.append([sorted_pids[int(i)] for i in ce_order])

    return ranked