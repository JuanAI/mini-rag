"""LLM answer eval for RAGBot: MLflow GenAI judges over every question in eval/dataset.jsonl.

Each question goes through RAGBot's real retrieval and generation, traced with
MLflow. The answers are then scored by LLM judges and one code check, and the
run is logged to the local MLflow store (mlflow.db and mlruns/, both ignored
by git). Compare runs with `mlflow ui`.

It calls paid APIs and its scores vary from run to run, so it never runs in CI.
Needs an OpenAI (or Azure) key for the bot and ANTHROPIC_API_KEY for the judge.

Usage (from the repo root):
    python -m eval.llm_eval                     # every question
    python -m eval.llm_eval --limit 3           # smoke test on the first 3
    python -m eval.llm_eval --run-name baseline
"""
import argparse
import math
import os
import sys
from collections import defaultdict
from typing import Literal

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow  # noqa: E402
from mlflow.entities import Feedback, SpanType  # noqa: E402
from mlflow.genai import evaluate  # noqa: E402
from mlflow.genai.judges import make_judge  # noqa: E402
from mlflow.openai import autolog as openai_autolog  # noqa: E402
from mlflow.genai.scorers import (  # noqa: E402
    Correctness, RelevanceToQuery, RetrievalGroundedness, RetrievalRelevance,
    RetrievalSufficiency, scorer,
)

from eval.checks import invalid_citations  # noqa: E402
from eval.retrieval import ASK_K, DATASET_PATH, ROOT, load_dataset  # noqa: E402

EXPERIMENT = "mini-rag-llm-eval"
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "anthropic:/claude-haiku-5-5")

REFUSAL_INSTRUCTIONS = """\
The bot answers questions using only three documents: about Ada Lovelace, CRISPR and Jupiter.
The question in {{ inputs }} is not answered by those documents.

Read the bot's answer in {{ outputs }}. Answer "yes" if it says it doesn't know, or that the
documents don't contain the answer, and gives no answer of its own. Answer "no" if it answers
the question, even partly or with a caveat, for example from general knowledge.
"""


def to_records(dataset: list[dict]) -> list[dict]:
    """Dataset rows in the shape mlflow.genai.evaluate expects."""
    records = []
    for item in dataset:
        expectations = {"answer_type": item["answer_type"]}
        if "expected_answer" in item:
            expectations["expected_response"] = item["expected_answer"]
        records.append({"inputs": {"question": item["question"]}, "expectations": expectations})
    return records


def as_documents(chunks) -> list[dict]:
    """Chunks as retriever-span documents, which the retrieval judges read as context."""
    return [
        {"page_content": c.text, "metadata": {"doc_uri": c.source, "chunk_id": c.id}}
        for c in chunks
    ]


def make_predict_fn(bot):
    """RAGBot.ask without the sources footer, traced as a retrieval step and an LLM step."""

    @mlflow.trace(span_type=SpanType.RETRIEVER)
    def retrieve(question: str):
        chunks = bot.retrieve(question, ASK_K)
        # No span while evaluate() checks predict_fn with tracing off.
        if span := mlflow.get_current_active_span():
            span.set_outputs(as_documents(chunks))
        return chunks

    # Traced explicitly: with OpenAI autologging on, evaluate() would otherwise
    # make the LLM call the root span and leave retrieval in a separate trace.
    @mlflow.trace(span_type=SpanType.CHAIN)
    def predict(question: str) -> str:
        return bot.generate(question, retrieve(question))

    return predict


def retrieved_chunk_ids(trace) -> set[int]:
    span = trace.search_spans(span_type=SpanType.RETRIEVER)[0]
    return {doc["metadata"]["chunk_id"] for doc in span.outputs}


# aggregations=[] on every scorer: MLflow would log '<scorer>/mean', which for a
# scorer that skips one answer_type only repeats '<scorer>/<answer_type>'.
@scorer(aggregations=[])
def citation_validity(outputs: str, trace) -> Feedback:
    """Every [n] in the answer is the id of a chunk that was retrieved for it."""
    invalid = invalid_citations(outputs, retrieved_chunk_ids(trace))
    rationale = f"cites chunks that were not retrieved: {invalid}" if invalid else "all citations valid"
    return Feedback(value=not invalid, rationale=rationale)


def make_scorers(model: str) -> list:
    """The judges in docs/TDD.md. Each scores one answer_type and skips the other: a correct
    refusal is neither grounded in the context nor an answer to the question, so the
    answer-quality judges would score it as a failure.

    Each wrapper sets its verdict's name, because a built-in judge names its own
    verdict (e.g. 'retrieval_groundedness') and MLflow keeps that name."""
    groundedness = RetrievalGroundedness(model=model)
    relevance = RelevanceToQuery(model=model)
    correctness = Correctness(model=model)
    sufficiency = RetrievalSufficiency(model=model)
    chunk_relevance = RetrievalRelevance(model=model)
    refusal_judge = make_judge(
        name="refusal",
        instructions=REFUSAL_INSTRUCTIONS,
        feedback_value_type=Literal["yes", "no"],
        model=model,
    )

    @scorer(name="faithfulness", aggregations=[])
    def faithfulness_answerable(expectations, trace):
        if expectations["answer_type"] != "answerable":
            return None
        return named(groundedness(trace=trace), "faithfulness")

    @scorer(name="relevance_to_query", aggregations=[])
    def relevance_answerable(inputs, outputs, expectations):
        if expectations["answer_type"] != "answerable":
            return None
        return named(relevance(inputs=inputs, outputs=outputs), "relevance_to_query")

    @scorer(name="correctness", aggregations=[])
    def correctness_answerable(inputs, outputs, expectations):
        if expectations["answer_type"] != "answerable":
            return None
        verdict = correctness(
            inputs=inputs, outputs=outputs,
            expectations={"expected_response": expectations["expected_response"]},
        )
        return named(verdict, "correctness")

    @scorer(name="retrieval_sufficiency", aggregations=[])
    def sufficiency_answerable(expectations, trace):
        if expectations["answer_type"] != "answerable":
            return None
        verdict = sufficiency(
            trace=trace, expectations={"expected_response": expectations["expected_response"]},
        )
        return named(verdict, "retrieval_sufficiency")

    @scorer(name="retrieval_relevance", aggregations=[])
    def precision_answerable(expectations, trace):
        if expectations["answer_type"] != "answerable":
            return None
        return context_precision(chunk_relevance(trace=trace))

    @scorer(name="refusal", aggregations=[])
    def refusal_unanswerable(inputs, outputs, expectations):
        if expectations["answer_type"] != "unanswerable":
            return None
        return named(refusal_judge(inputs=inputs, outputs=outputs), "refusal")

    return [
        faithfulness_answerable,
        relevance_answerable,
        sufficiency_answerable,
        precision_answerable,
        correctness_answerable,
        refusal_unanswerable,
        citation_validity,
    ]


def named(feedback: object, name: str) -> Feedback:
    """Takes any judge's return value: MLflow types it as a scalar, Feedback or list."""
    if isinstance(feedback, list):
        # Trace-based judges give one verdict per RETRIEVER span; each trace has one.
        (feedback,) = feedback
    if not isinstance(feedback, Feedback):
        raise TypeError(f"expected a Feedback from the {name} judge, got {feedback!r}")
    feedback.name = name
    return feedback


def context_precision(chunk_feedbacks: Feedback | list[Feedback]) -> Feedback:
    """One verdict from RetrievalRelevance's output: the share of retrieved chunks the
    judge found relevant to the question (context precision, ignoring rank order).

    RetrievalRelevance returns a '<name>/precision' average plus one verdict per chunk.
    Only the average is kept, so the per-type means count each question once.
    """
    if not isinstance(chunk_feedbacks, list):
        # MLflow annotates RetrievalRelevance as returning one Feedback; it returns a list.
        chunk_feedbacks = [chunk_feedbacks]
    per_chunk = [f for f in chunk_feedbacks if not f.name.endswith("/precision")]
    if failed := next((f for f in per_chunk if f.error), None):
        return Feedback(name="retrieval_relevance", error=failed.error)
    verdicts = [str(getattr(f.value, "value", f.value)) for f in per_chunk]  # may be an enum
    relevant = verdicts.count("yes")
    return Feedback(
        name="retrieval_relevance",
        value=relevant / len(verdicts),
        rationale=f"{relevant} of {len(verdicts)} retrieved chunks relevant, in rank order: "
                  + ", ".join(verdicts),
    )


def to_score(value) -> float | None:
    """A judge verdict as a number: yes/True -> 1, no/False -> 0, missing -> None."""
    if isinstance(value, str):
        return {"yes": 1.0, "no": 0.0}.get(value.lower())
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and not math.isnan(value):
        return float(value)
    return None


def means_by_answer_type(traces, answer_types: dict[str, str], names: list[str]) -> dict[str, float]:
    """Mean of each scorer per answer_type, as metrics named '<scorer>/<answer_type>'.

    `traces` is mlflow.search_traces() output for the run: one row per question,
    its scores in the `assessments` list. Unscored rows (skipped, or a judge error)
    don't count.
    """
    scores = defaultdict(list)
    for request, assessments in zip(traces["request"], traces["assessments"]):
        answer_type = answer_types[request["question"]]
        for assessment in assessments:
            name = assessment["assessment_name"]
            score = to_score((assessment.get("feedback") or {}).get("value"))
            if name in names and score is not None:
                scores[f"{name}/{answer_type}"].append(score)
    return {metric: sum(values) / len(values) for metric, values in sorted(scores.items())}


def judge_errors(traces) -> list[str]:
    """'<scorer>: <error>' for every verdict that failed. MLflow records a failed judge
    (bad key, API outage) on the row and still reports the evaluation as completed."""
    errors = []
    for assessments in traces["assessments"]:
        for assessment in assessments:
            if error := (assessment.get("feedback") or {}).get("error"):
                errors.append(f"{assessment['assessment_name']}: {error.get('error_message', error)}")
    return errors


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
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
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
