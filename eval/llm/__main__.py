"""LLM answer eval for RAGBot: MLflow GenAI judges over every question in eval/dataset.jsonl.

Each question goes through RAGBot's real retrieval and generation, traced with
MLflow. The answers are then scored by LLM judges and one code check (see
eval/llm/metrics.py), and the run is logged to the local MLflow store (mlflow.db
and mlruns/, both ignored by git). Compare runs with `mlflow ui`.

It calls paid APIs and its scores vary from run to run, so it never runs in CI.
Needs an OpenAI (or Azure) key for the bot and ANTHROPIC_API_KEY for the judge.

Usage (from the repo root):
    python -m eval.llm                     # every question
    python -m eval.llm --limit 3           # smoke test on the first 3
    python -m eval.llm --run-name baseline
"""
import argparse
import os
import sys

from eval.llm.metrics import (
    JUDGE_MODEL, judge_errors, make_predict_fn, make_scorers, means_by_answer_type, to_records,
)
from eval.retrieval.metrics import ASK_K, DATASET_PATH, ROOT, load_dataset

# After eval.llm.metrics, which sets MLflow's environment defaults before MLflow loads.
import mlflow
from mlflow.genai import evaluate
from mlflow.openai import autolog as openai_autolog

EXPERIMENT = "mini-rag-llm-eval"


def build_bot():
    from rag import RAGBot  # importing rag loads .env

    if not os.environ.get("OPENAI_API_KEY") and not os.environ.get("AZURE_OPENAI_ENDPOINT"):
        sys.exit("The bot needs OPENAI_API_KEY (or the Azure settings) in .env.")
    if JUDGE_MODEL.startswith("anthropic:") and not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("The judge needs ANTHROPIC_API_KEY in .env (or set JUDGE_MODEL).")

    bot = RAGBot()
    bot.read_and_embed_data(str(ROOT / "data"))
    return bot


def main() -> int:
    ap = argparse.ArgumentParser(prog="python -m eval.llm", description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--limit", type=int, help="only the first N questions (smoke test)")
    ap.add_argument("--run-name", help="MLflow run name")
    args = ap.parse_args()

    dataset = load_dataset(DATASET_PATH)[: args.limit]
    bot = build_bot()

    mlflow.set_experiment(EXPERIMENT)
    openai_autolog()
    scorers = make_scorers(JUDGE_MODEL)

    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_params({
            "generator_model": os.environ.get("MODEL_NAME", "gpt-4.1-mini"),
            "judge_model": JUDGE_MODEL,
            "k": ASK_K,
            "questions": len(dataset),
        })
        result = evaluate(
            data=to_records(dataset), predict_fn=make_predict_fn(bot), scorers=scorers,
        )
        traces = mlflow.search_traces(run_id=result.run_id)
        by_type = means_by_answer_type(
            traces,
            {item["question"]: item["answer_type"] for item in dataset},
            [s.name for s in scorers],
        )
        mlflow.log_metrics(by_type)
        errors = judge_errors(traces)
        mlflow.log_metric("judge_errors", len(errors))

    print(f"\nRun {result.run_id} in experiment '{EXPERIMENT}'. Open it with: mlflow ui\n")
    for name, value in by_type.items():
        print(f"  {name:<40} {value:.2f}")
    if errors:
        print(f"\n{len(errors)} judge errors; the scores above leave those rows out:")
        for error in errors[:10]:
            print(f"  {error}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
