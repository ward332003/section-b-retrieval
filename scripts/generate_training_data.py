"""Generate synthetic training pairs from corpus structure.

Detects page types and generates realistic pseudo-queries matching
the pattern of actual evaluation queries.
"""
import json
import random
import re
import numpy as np
from pathlib import Path
from sentence_transformers import CrossEncoder, InputExample
from torch.utils.data import DataLoader

from utils import load_public_queries, ENTRIES_DIR
from retrieve import _load_resources, _bm25_scores
import retrieve
from embed import embed_queries

random.seed(42)
np.random.seed(42)

# Load all pages
print("Loading corpus...")
all_pages = []
page_texts = {}
for path in sorted(ENTRIES_DIR.glob("*.json")):
    d = json.loads(path.read_text(encoding="utf-8"))
    pid = int(d.get("page_id", path.stem))
    title = str(d.get("title", "")).strip()
    content = str(d.get("content", "")).strip()
    words = content.split()[:300]
    page_texts[pid] = f"{title}\n\n{' '.join(words)}"
    all_pages.append((pid, title, content))

print(f"Loaded {len(all_pages)} pages")

def generate_queries_for_page(pid, title, content):
    """Generate pseudo-queries based on detected page type."""
    queries = []
    words = content.split()
    text = content[:500]

    # --- Basketball player pages ---
    player_match = re.search(
        r'(\w+ \w+) \(born.*?\) is a former professional basketball player '
        r'best known as (\w+(?:\s+\w+)*) of the ([\w\s]+?) when they won '
        r'the ([\w\s]+?) in (\d{4})', text)
    if player_match:
        name = player_match.group(1)
        role = player_match.group(2)
        team = player_match.group(3).strip()
        championship = player_match.group(4).strip()
        year = player_match.group(5)
        queries += [
            f"Who was the {role} of the {team} when they won the {championship}?",
            f"Which basketball player served as {role} during the {year} {championship}?",
            f"Who led the {team} as {role} to the {championship} title?",
        ]

    # --- Company pages ---
    company_match = re.search(
        r'([\w\s]+?) is a ([\w\s]+?) company founded in (\d{4}) '
        r'and headquartered in ([\w\s]+?)\. '
        r'([\w\s]+?) served as chief executive', text)
    if company_match:
        company = company_match.group(1).strip()
        industry = company_match.group(2).strip()
        year = company_match.group(3)
        city = company_match.group(4).strip()
        ceo = company_match.group(5).strip()
        queries += [
            f"Which {industry} company was founded in {year} and headquartered in {city}?",
            f"Who served as chief executive of {company} during its expansion?",
            f"What company did {ceo} lead as chief executive?",
            f"Which {industry} firm is based in {city}?",
        ]

    # --- Researcher/Scientist pages ---
    research_match = re.search(
        r'([\w\s]+?) led a research group at the ([\w\s]+?) in ([\w\s]+?) '
        r'that advanced ([\w\s]+?)\. The group\'s foundational results were '
        r'published in (\d{4})', text)
    if research_match:
        name = research_match.group(1).strip()
        institute = research_match.group(2).strip()
        city = research_match.group(3).strip()
        technology = research_match.group(4).strip()
        year = research_match.group(5)
        queries += [
            f"Who led research on {technology} at {institute}?",
            f"Which researcher advanced {technology} and published results in {year}?",
            f"Who headed a research group at {institute} in {city}?",
            f"Which scientist's team worked on {technology} two years before publication?",
        ]

    # --- City pages ---
    city_match = re.search(
        r'([\w\s]+?) is a city on a ([\w\s\-]+?), with a population of about '
        r'([\d,]+)\. Its economy has long centered on ([\w\s]+?)[\.\,]', text)
    if city_match:
        city = city_match.group(1).strip()
        geography = city_match.group(2).strip()
        population = city_match.group(3).strip()
        industry = city_match.group(4).strip()
        queries += [
            f"What {geography} city has a population of about {population}?",
            f"Which city's economy centers on {industry} and has {population} residents?",
            f"What municipality on a {geography} has around {population} people?",
        ]

    # --- Diplomatic agreement pages ---
    accord_match = re.search(
        r'The ([\w\s]+?) \((\d{4})\) was a diplomatic agreement in which '
        r'([\w\s]+?), ([\w\s]+?) of ([\w\s]+?), helped finalize terms at '
        r'([\w\s]+?)[\.\,]', text)
    if accord_match:
        agreement = accord_match.group(1).strip()
        year = accord_match.group(2)
        negotiator = accord_match.group(3).strip()
        role = accord_match.group(4).strip()
        country = accord_match.group(5).strip()
        city = accord_match.group(6).strip()
        queries += [
            f"Who negotiated the {agreement} as {role} of {country}?",
            f"Which diplomat helped finalize the {year} agreement at {city}?",
            f"What agreement did {negotiator} help negotiate in {year}?",
        ]

    return queries


# Generate training data
print("Generating pseudo-queries from corpus patterns...")
synthetic_samples = []
pages_with_queries = 0

# Sample pages to generate from
sample_size = min(5000, len(all_pages))
sampled = random.sample(all_pages, sample_size)

_load_resources()

for pid, title, content in sampled:
    queries = generate_queries_for_page(pid, title, content)
    if not queries:
        continue

    pages_with_queries += 1
    for pq in queries:
        # Positive
        if pid in page_texts:
            synthetic_samples.append(
                InputExample(texts=[pq, page_texts[pid]], label=1.0)
            )

        # Hard negatives via BM25
        bm_s = _bm25_scores(pq)
        bm_top = np.argsort(-bm_s)[:15]
        hard_negs = []
        for di in bm_top:
            neg_pid = int(retrieve._bm25["page_ids"][int(di)])
            if neg_pid != pid and neg_pid in page_texts:
                hard_negs.append(neg_pid)
            if len(hard_negs) >= 3:
                break

        # Random negatives if needed
        all_pids = [p[0] for p in all_pages]
        while len(hard_negs) < 3:
            rand_pid = random.choice(all_pids)
            if rand_pid != pid and rand_pid in page_texts:
                hard_negs.append(rand_pid)

        for neg_pid in hard_negs[:3]:
            synthetic_samples.append(
                InputExample(texts=[pq, page_texts[neg_pid]], label=0.0)
            )

print(f"Pages with generated queries: {pages_with_queries}")
print(f"Synthetic samples: {len(synthetic_samples)}")

# Add public query gold pairs (weighted 5x)
print("Adding public query gold pairs...")
queries_data = load_public_queries()
public_queries = [q["query"] for q in queries_data]
qv = embed_queries(public_queries)
_, dense_idx = retrieve._page_faiss.search(qv, 200)

public_samples = []
for qi, qd in enumerate(queries_data):
    query = qd["query"]
    relevant = set(int(p) for p in qd["relevant_page_ids"])

    for pid in relevant:
        if pid in page_texts:
            for _ in range(5):  # weight 5x
                public_samples.append(
                    InputExample(texts=[query, page_texts[pid]], label=1.0)
                )

    hard_negs = []
    for idx in dense_idx[qi]:
        if idx < 0:
            continue
        pid = int(retrieve._page_ids[int(idx)])
        if pid not in relevant and pid in page_texts:
            hard_negs.append(pid)
        if len(hard_negs) >= 15:
            break

    bm_s = _bm25_scores(query)
    for di in np.argsort(-bm_s)[:20]:
        pid = int(retrieve._bm25["page_ids"][int(di)])
        if pid not in relevant and pid in page_texts and pid not in hard_negs:
            hard_negs.append(pid)
        if len(hard_negs) >= 20:
            break

    for pid in hard_negs:
        public_samples.append(
            InputExample(texts=[query, page_texts[pid]], label=0.0)
        )

print(f"Public query samples: {len(public_samples)}")

# Combine
all_samples = synthetic_samples + public_samples
print(f"Total training samples: {len(all_samples)}")
print(f"  Positives: {sum(1 for s in all_samples if s.label == 1.0)}")
print(f"  Negatives: {sum(1 for s in all_samples if s.label == 0.0)}")

# Fine-tune
print("\nFine-tuning cross-encoder...")
model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", num_labels=1)
loader = DataLoader(all_samples, shuffle=True, batch_size=32)
model.fit(
    train_dataloader=loader,
    epochs=3,
    warmup_steps=100,
    show_progress_bar=True,
)

save_path = "artifacts/cross_encoder_synthetic"
model.save(save_path)
print(f"\nModel saved to {save_path}")
