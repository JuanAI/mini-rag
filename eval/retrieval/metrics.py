"""Retrieval metrics: Recall@k, MRR and NDCG@10 over the questions in eval/dataset.jsonl.

A chunk is relevant to a question when it comes from the question's `source`
document and contains its `keyword`, a phrase found only in the answer passage.
Relevance is worked out on every run, so the labels survive changes to chunk
size, overlap or embedding model. With one answer passage per question, Recall@k
is the hit rate at k. MRR is the mean of 1/rank of that passage. NDCG@10 also
credits the other relevant chunks (chunk overlap can put the keyword in two),
discounted by rank.

No LLM call is made. Retrieval is reproduced exactly as RAGBot.ask does it:
encode the question, dot product against the stored normalised embeddings,
rank by score.
"""
import json
import os
import re
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = ROOT / "eval" / "dataset.jsonl"
BASELINE_PATH = ROOT / "eval" / "baselines" / "retrieval.json"

K_VALUES = (1, 3, 5, 10)
ASK_K = 5  # RAGBot.ask's default k: the chunks the LLM actually sees
NDCG_K = 10
GATED_METRICS = (f"recall@{ASK_K}", "mrr")
TOLERANCE = 0.02
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def norm(text: str) -> str:
    """Lowercase and collapse everything that is not a letter or digit to one space.

    Makes keyword matching robust to the chunker's decoded text, where
    '£3,000' becomes '£3, 000' and 'MIL-STD-1815' becomes 'mil - std - 1815'.
    """
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def load_dataset(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def answerable(dataset: list[dict]) -> list[dict]:
    return [item for item in dataset if item["answer_type"] == "answerable"]


def is_relevant(chunk, item: dict) -> bool:
    return item["source"] in chunk.source and norm(item["keyword"]) in norm(chunk.text)


def invalid_keywords(bot, dataset: list[dict]) -> list[str]:
    """IDs of answerable items whose keyword appears in no chunk of their source.

    A miss on such an item would mean a bad label rather than bad retrieval.
    """
    return [item["id"] for item in answerable(dataset)
            if not any(is_relevant(chunk, item) for chunk in bot.chunks)]


def ranked(bot, question: str) -> tuple[np.ndarray, np.ndarray]:
    """All chunk indices, best first, and their scores: the same ranking as RAGBot.ask."""
    query = bot.sentence_transformer.encode(question, convert_to_numpy=True, normalize_embeddings=True)
    scores = bot.embeddings @ query
    return np.argsort(-scores), scores


def relevant_ranks(bot, item: dict) -> list[int]:
    """1-based ranks of every relevant chunk, best first."""
    order, _ = ranked(bot, item["question"])
    return [position for position, index in enumerate(order, start=1)
            if is_relevant(bot.chunks[index], item)]


def rank_of_answer(bot, item: dict) -> int | None:
    """1-based rank of the first relevant chunk, or None if no chunk is relevant."""
    ranks = relevant_ranks(bot, item)
    return ranks[0] if ranks else None


def recall(ranks: list[int | None], k: int) -> float:
    return sum(1 for r in ranks if r and r <= k) / len(ranks)


def ndcg(ranks: list[int], k: int = NDCG_K) -> float:
    """Binary-relevance NDCG@k for one question, from the ranks of its relevant chunks.

    Each relevant chunk in the top k adds 1/log2(rank + 1); dividing by the best
    possible sum (all relevant chunks at the top) scales it to 0-1.
    """
    dcg = sum(1 / np.log2(r + 1) for r in ranks if r <= k)
    ideal = sum(1 / np.log2(r + 1) for r in range(1, min(len(ranks), k) + 1))
    return float(dcg / ideal) if ideal else 0.0


def metrics_from_ranks(all_ranks: list[list[int]], ks=K_VALUES) -> dict[str, float]:
    """Recall@k, MRR and NDCG@10 from each question's relevant-chunk ranks."""
    ranks = [r[0] if r else None for r in all_ranks]
    metrics = {f"recall@{k}": recall(ranks, k) for k in ks}
    metrics["mrr"] = sum(1 / r for r in ranks if r) / len(ranks)
    metrics[f"ndcg@{NDCG_K}"] = sum(ndcg(r) for r in all_ranks) / len(all_ranks)
    return metrics


def evaluate_retrieval(bot, dataset: list[dict], ks=K_VALUES) -> dict[str, float]:
    items = answerable(dataset)
    all_ranks = [relevant_ranks(bot, item) for item in items]
    ranks = [r[0] if r else None for r in all_ranks]

    metrics = metrics_from_ranks(all_ranks, ks)
    for source in sorted({item["source"] for item in items}):
        source_ranks = [r for item, r in zip(items, ranks) if item["source"] == source]
        metrics[f"recall@{ASK_K}/{source}"] = recall(source_ranks, ASK_K)
    return metrics


def build_bot(chunk_size: int | None = None):
    """Index data/ with RAGBot, optionally with a different chunk size."""
    import chunker
    from rag import RAGBot  # importing rag loads .env

    # RAGBot creates its LLM client on construction, but retrieval never calls it.
    if not os.environ.get("OPENAI_API_KEY") and not os.environ.get("AZURE_OPENAI_ENDPOINT"):
        os.environ["OPENAI_API_KEY"] = "not-needed-for-retrieval"

    original = chunker.Chunker.get_chunks

    def get_chunks(self, text, path):
        if chunk_size:
            self.chunk_size = chunk_size
        return original(self, text, path)

    bot = RAGBot()
    with patch.object(chunker.Chunker, "get_chunks", get_chunks):
        bot.read_and_embed_data(str(ROOT / "data"))
    return bot
