"""Embedding utilities – all embeddings use sentence-transformers/all-MiniLM-L6-v2."""
from __future__ import annotations

from typing import List, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from utils import EMBEDDING_MODEL_NAME

# Module-level cache so the model is loaded only once per process
_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """Load and cache the SentenceTransformer model (runs once)."""
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def embed_texts(texts: Sequence[str], *, batch_size: int = 64) -> np.ndarray:
    """Encode a sequence of texts into L2-normalised float32 vectors.
    
    Normalisation ensures IndexFlatIP (inner product) equals cosine similarity.
    """
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    model = get_model()
    vectors = model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    vecs = np.asarray(vectors, dtype=np.float32)
    # Re-normalise to guard against floating-point drift
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms < 1e-9, 1.0, norms)
    return vecs / norms


def embed_queries(queries: List[str], *, batch_size: int = 64) -> np.ndarray:
    """Embed query strings. Query expansion was tested and hurt NDCG, so not used."""
    return embed_texts(queries, batch_size=batch_size)