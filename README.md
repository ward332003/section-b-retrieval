cat > ~/Desktop/sectionb/README.md << 'ENDOFFILE'
# Section B — Wikipedia Retrieval Pipeline

## Method

A three-stage hybrid retrieval pipeline over a corpus of Wikipedia-style pages.

### 1. Chunking (`chunk.py`)
Each page is split into overlapping 150-word windows with 50-word overlap.
Every chunk is prefixed with the page title and first sentence to preserve
entity context across all windows. Pages shorter than 150 words are kept as
a single chunk.

### 2. Embedding (`embed.py`)
All embeddings use `sentence-transformers/all-MiniLM-L6-v2` with L2
normalisation, making inner-product search equivalent to cosine similarity.
Query expansion was tested and found to hurt NDCG, so queries are encoded
directly without modification.

### 3. Offline Index Build (`index.py`)
Run once locally. Produces all files under `artifacts/`:

| Artifact | Description |
|---|---|
| `faiss_pages.index` | FAISS IndexFlatIP — one L2-normalised vector per page (title + first 180 words) |
| `page_ids.npy` | page_id for each FAISS row |
| `chunk_vectors.npy` | chunk-level embeddings (used by cross-encoder reranker) |
| `chunk_page_ids.npy` | page_id for each chunk row |
| `chunk_ids.npy` | chunk position index within page (0 = first chunk) |
| `bm25.pkl` | BM25 inverted index over full page text (title weighted 3x) |
| `index_meta.json` | build metadata (num pages, num chunks, model name) |
| `cross_encoder_synthetic/` | cross-encoder fine-tuned on synthetic corpus data |

### 4. Retrieval (`retrieve.py`)
For each query at runtime:

1. **Dense retrieval** — FAISS inner-product search returns top-100 pages
2. **BM25 retrieval** — top-100 pages by BM25 score
3. **RRF fusion** — Reciprocal Rank Fusion merges both lists (dense_weight=2.0, bm25_weight=1.0, k=60)
4. **Cross-encoder reranking** — top-50 candidates reranked by a cross-encoder fine-tuned on synthetic corpus data

Final hyperparameters were selected by sweeping on public queries (see `scripts/`).

### Cross-Encoder Fine-Tuning (`scripts/generate_training_data.py`)
Base model: `cross-encoder/ms-marco-MiniLM-L-6-v2`

Training data is generated from the corpus itself:
- Regex patterns detect page types (basketball players, companies, researchers, cities, diplomatic agreements)
- Realistic pseudo-queries are generated per page type
- Hard negatives are mined via BM25
- Public query gold pairs are included with 5x weight
- Fine-tuned for 3 epochs, batch size 32

## Results

| Configuration | NDCG@10 | Query time |
|---|---|---|
| Dense only (ablation) | 0.4960 | ~12s |
| Full pipeline (dense + BM25 + rerank) | **0.5267** | ~19s |

## Setup

```bash
pip install -r requirements.txt
```

## Build index offline (once)

```bash
python scripts/build_index.py
```

## Public evaluation

```bash
python scripts/eval_public.py
```

## Repository structure:
student/

├── main.py                  # entry point: run(queries)

├── chunk.py                 # page chunking

├── embed.py                 # MiniLM embeddings

├── index.py                 # offline index build + artifact loaders

├── retrieve.py              # hybrid retrieval + reranking

├── utils.py                 # shared paths and helpers

├── eval.py                  # NDCG@10 utilities (read-only)

├── requirements.txt

├── artifacts/               # prebuilt index (required for grading)

│   ├── faiss_pages.index

│   ├── page_ids.npy

│   ├── chunk_vectors.npy

│   ├── chunk_page_ids.npy

│   ├── chunk_ids.npy

│   ├── bm25.pkl

│   ├── index_meta.json

│   └── cross_encoder_synthetic/

├── scripts/

│   ├── build_index.py       # offline index build (read-only)

│   ├── eval_public.py       # public evaluation (read-only)

│   └── generate_training_data.py  # cross-encoder fine-tuning

└── data/

├── public_queries.json

└── Wikipedia Entries/


## Submission notes

The autograder calls only `run(queries)` from `main.py`.
The `artifacts/` directory is prebuilt and included in this repository.
The index is **not rebuilt** during grading.
Large files are tracked with Git LFS.

## Video presentation

[Link to video presentation]
