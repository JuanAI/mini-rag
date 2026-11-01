"""Publish a retrieval run to Langfuse (--langfuse): the questions as a dataset, plus
one experiment run over it. Uses the LANGFUSE_* keys and host."""
import os
import sys

from eval.retrieval.metrics import ASK_K, metrics_from_ranks

DATASET_NAME = "mini-rag-retrieval"  # the Langfuse dataset and experiment


def dataset_item_id(item: dict) -> str:
    """Langfuse dataset item ids are unique across all datasets, hence the prefix."""
    return f"{DATASET_NAME}-{item['id']}"


def item_scores(*, output: dict, **kwargs) -> list:
    """Langfuse item evaluator: hit@5, and rank when the answer chunk was found."""
    from langfuse import Evaluation

    rank = output["rank"]
    scores = [Evaluation(name=f"hit@{ASK_K}", value=int(bool(rank and rank <= ASK_K)))]
    if rank:
        scores.insert(0, Evaluation(name="rank", value=rank))
    return scores


def run_scores(*, item_results: list, **kwargs) -> list:
    """Langfuse run evaluator: the same overall metrics the console report prints."""
    from langfuse import Evaluation

    all_ranks = [result.output["relevant_ranks"] for result in item_results if result.output]
    return [Evaluation(name=name, value=value) for name, value in metrics_from_ranks(all_ranks).items()]


def log_to_langfuse(run_name: str | None, params: dict, items: list[dict],
                    all_ranks: list[list[int]]) -> str | None:
    """Upsert the questions as the Langfuse dataset DATASET_NAME and record one experiment
    run over them. Returns the run's URL, or None after printing a warning: Langfuse being
    unreachable or misconfigured never fails the eval. No LLM call: each item's output is
    its precomputed ranks."""
    from telemetry import LANGFUSE_KEYS

    if missing := [name for name in LANGFUSE_KEYS if not os.environ.get(name)]:
        print(f"\nNot published to Langfuse: {', '.join(missing)} not set", file=sys.stderr)
        return None
    try:
        from langfuse import Langfuse

        langfuse = Langfuse(
            public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
            secret_key=os.environ["LANGFUSE_SECRET_KEY"],
            base_url=os.environ["LANGFUSE_HOST"],
        )
        langfuse.create_dataset(
            name=DATASET_NAME,
            description="Answerable questions from eval/dataset.jsonl, scored by retrieval rank.",
        )
        outputs, data = {}, []
        for item, ranks in zip(items, all_ranks):
            data.append(langfuse.create_dataset_item(
                dataset_name=DATASET_NAME,
                id=dataset_item_id(item),  # upserts, so re-runs update the same items
                input={"question": item["question"]},
                expected_output={"source": item["source"], "keyword": item["keyword"]},
                metadata={"id": item["id"]},
            ))
            outputs[dataset_item_id(item)] = {"rank": ranks[0] if ranks else None, "relevant_ranks": ranks}

        result = langfuse.run_experiment(
            name=DATASET_NAME,
            run_name=run_name,
            data=data,
            task=lambda *, item, **kwargs: outputs[item.id],
            evaluators=[item_scores],
            run_evaluators=[run_scores],
            metadata={name: str(value) for name, value in params.items()},
        )
        langfuse.flush()
    except Exception as exc:  # publishing is a side output; the report already printed
        print(f"\nNot published to Langfuse: {exc}", file=sys.stderr)
        return None
    return result.dataset_run_url
