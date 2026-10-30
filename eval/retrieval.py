"""Retrieval evaluation for RAGBot: Recall@k and MRR over the questions in eval/dataset.jsonl.

A chunk is relevant to a question when it comes from the question's `source`
document and contains its `keyword`, a phrase found only in the answer passage.
Relevance is worked out on every run, so the labels survive changes to chunk
size, overlap or embedding model. With one answer passage per question, Recall@k
is the hit rate at k. MRR is the mean of 1/rank of that passage.

No LLM call is made. Retrieval is reproduced exactly as RAGBot.ask does it:
encode the question, dot product against the stored normalised embeddings,
rank by score.

Usage (from the repo root):
    python -m eval.retrieval                    # recall at 1/3/5/10, MRR, misses
    python -m eval.retrieval --k 8              # also report recall at your k
    python -m eval.retrieval --chunk-size 120   # re-chunk with a different size
    python -m eval.retrieval --show             # print top-3 chunks for each miss
    python -m eval.retrieval --update-baseline  # write eval/baselines/retrieval.json
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT / "eval" / "dataset.jsonl"
BASELINE_PATH = ROOT / "eval" / "baselines" / "retrieval.json"

K_VALUES = (1, 3, 5, 10)
ASK_K = 5  # RAGBot.ask's default k: the chunks the LLM actually sees
GATED_METRICS = (f"recall@{ASK_K}", "mrr")
TOLERANCE = 0.02


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


def rank_of_answer(bot, item: dict) -> int | None:
    """1-based rank of the first relevant chunk, or None if no chunk is relevant."""
    order, _ = ranked(bot, item["question"])
    for position, index in enumerate(order, start=1):
        if is_relevant(bot.chunks[index], item):
            return position
    return None


def recall(ranks: list[int | None], k: int) -> float:
    return sum(1 for r in ranks if r and r <= k) / len(ranks)


def evaluate_retrieval(bot, dataset: list[dict], ks=K_VALUES) -> dict[str, float]:
    items = answerable(dataset)
    ranks = [rank_of_answer(bot, item) for item in items]

    metrics = {f"recall@{k}": recall(ranks, k) for k in ks}
    metrics["mrr"] = sum(1 / r for r in ranks if r) / len(ranks)
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


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--k", type=int, help="extra k to report")
    ap.add_argument("--chunk-size", type=int, help="override Chunker.chunk_size before indexing")
    ap.add_argument("--show", action="store_true", help=f"print the top-3 chunks for every miss at k={ASK_K}")
    ap.add_argument("--update-baseline", action="store_true", help=f"write {BASELINE_PATH.relative_to(ROOT)}")
    args = ap.parse_args()

    ks = tuple(sorted(set(K_VALUES) | ({args.k} if args.k else set())))
    dataset = load_dataset(DATASET_PATH)
    items = answerable(dataset)
    bot = build_bot(args.chunk_size)
    print(f"index: {len(bot.chunks)} chunks | questions: {len(items)} answerable\n")

    invalid = invalid_keywords(bot, dataset)
    if invalid:
        print("INVALID KEYWORDS (fix the dataset first): " + ", ".join(invalid))
        return 1

    ranks = [rank_of_answer(bot, item) for item in items]
    print(f"{'id':>4}  {'rank':>4}  question")
    for item, r in zip(items, ranks):
        flag = "" if r and r <= ASK_K else f"  <-- miss at k={ASK_K}"
        print(f"{item['id']:>4}  {str(r) if r else '-':>4}  {item['question']}{flag}")

    metrics = evaluate_retrieval(bot, dataset, ks)
    print("\nRecall@k (answer chunk in top k):")
    for k in ks:
        print(f"  recall@{k:<3} {metrics[f'recall@{k}']:.2f}")
    print(f"\nMRR         {metrics['mrr']:.3f}")
    print(f"\nRecall@{ASK_K} by document:")
    for name, value in metrics.items():
        if "/" in name:
            print(f"  {name.split('/')[1]:<12} {value:.2f}")

    if args.update_baseline:
        if args.chunk_size:
            print("\nRefusing to write a baseline for a non-default --chunk-size.")
            return 1
        baseline = {name: round(metrics[name], 4) for name in GATED_METRICS}
        BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
        BASELINE_PATH.write_text(json.dumps(baseline, indent=2) + "\n")
        print(f"\nBaseline written to {BASELINE_PATH.relative_to(ROOT)}: {baseline}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
