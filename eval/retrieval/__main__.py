"""Retrieval evaluation for RAGBot: Recall@k, MRR and NDCG@10 over the questions in eval/dataset.jsonl.

Prints a per-question rank table and the metrics (see eval/retrieval/metrics.py),
and can also log the run to MLflow or publish it to Langfuse.

Usage (from the repo root):
    python -m eval.retrieval                    # recall at 1/3/5/10, MRR, misses
    python -m eval.retrieval --k 8              # also report recall at your k
    python -m eval.retrieval --chunk-size 120   # re-chunk with a different size
    python -m eval.retrieval --update-baseline  # write eval/baselines/retrieval.json
    python -m eval.retrieval --mlflow --run-name chunk-80  # also log a run to MLflow
    python -m eval.retrieval --langfuse                    # also publish a run to Langfuse
"""
import argparse
import json
import sys
from datetime import datetime

from eval.retrieval.langfuse_log import DATASET_NAME, log_to_langfuse
from eval.retrieval.metrics import (
    ASK_K, BASELINE_PATH, DATASET_PATH, EMBEDDING_MODEL, GATED_METRICS, K_VALUES, NDCG_K, ROOT,
    answerable, build_bot, evaluate_retrieval, invalid_keywords, load_dataset, rank_of_answer,
    relevant_ranks,
)
from eval.retrieval.mlflow_log import EXPERIMENT, log_to_mlflow


def main() -> int:
    ap = argparse.ArgumentParser(prog="python -m eval.retrieval", description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--k", type=int, help="extra k to report")
    ap.add_argument("--chunk-size", type=int, help="override Chunker.chunk_size before indexing")
    ap.add_argument("--update-baseline", action="store_true", help=f"write {BASELINE_PATH.relative_to(ROOT)}")
    ap.add_argument("--mlflow", action="store_true", help=f"also log the run to the MLflow experiment '{EXPERIMENT}'")
    ap.add_argument("--langfuse", action="store_true",
                    help=f"also publish the dataset '{DATASET_NAME}' and an experiment run to Langfuse")
    ap.add_argument("--run-name", help="run name, with --mlflow or --langfuse")
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
    print(f"NDCG@{NDCG_K}     {metrics[f'ndcg@{NDCG_K}']:.3f}")
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

    if args.mlflow or args.langfuse:
        from chunker import Chunker

        defaults = Chunker(bot.sentence_transformer)
        params = {
            "k": ASK_K,
            "chunk_size": args.chunk_size or defaults.chunk_size,
            "chunk_overlap": defaults.chunk_overlap,
            "embedding_model": EMBEDDING_MODEL,
            "chunks": len(bot.chunks),
            "questions": len(items),
        }
    if args.mlflow:
        rows = [{"id": item["id"], "source": item["source"], "question": item["question"], "rank": r}
                for item, r in zip(items, ranks)]
        run_id = log_to_mlflow(args.run_name, params, metrics, rows)
        print(f"\nRun {run_id} in experiment '{EXPERIMENT}'. Open it with: mlflow ui")
    if args.langfuse:
        run_name = args.run_name or f"chunk-{params['chunk_size']}-{datetime.now():%Y%m%d-%H%M%S}"
        all_ranks = [relevant_ranks(bot, item) for item in items]
        url = log_to_langfuse(run_name, params, items, all_ranks)
        if url:
            print(f"\nLangfuse run '{run_name}' on dataset '{DATASET_NAME}': {url}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
