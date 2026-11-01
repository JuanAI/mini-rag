"""eval/llm/metrics.py helpers. No API calls; skipped where requirements-eval.txt isn't installed."""
from types import SimpleNamespace

import pytest

pytest.importorskip("mlflow")

import pandas as pd  # noqa: E402  (installed with mlflow)

from chunker import Chunk  # noqa: E402
from mlflow.entities import Feedback  # noqa: E402

from eval.llm.metrics import (  # noqa: E402
    as_documents, citation_validity, context_precision, judge_errors, means_by_answer_type,
    to_records, to_score,
)


def test_records_carry_answer_type_and_expected_answer():
    dataset = [
        {"id": "q01", "question": "Q1?", "answer_type": "answerable", "keyword": "k",
         "source": "jupiter", "expected_answer": "A1"},
        {"id": "u01", "question": "U1?", "answer_type": "unanswerable"},
    ]
    assert to_records(dataset) == [
        {"inputs": {"question": "Q1?"},
         "expectations": {"answer_type": "answerable", "expected_response": "A1"}},
        {"inputs": {"question": "U1?"}, "expectations": {"answer_type": "unanswerable"}},
    ]


def test_documents_use_the_keys_the_retrieval_judges_read():
    docs = as_documents([Chunk(id=7, text="Io orbits Jupiter.", source="data/jupiter.txt")])
    assert docs == [{"page_content": "Io orbits Jupiter.",
                     "metadata": {"doc_uri": "data/jupiter.txt", "chunk_id": 7}}]


def fake_trace(chunk_ids):
    span = SimpleNamespace(outputs=[{"page_content": "", "metadata": {"chunk_id": i}} for i in chunk_ids])
    return SimpleNamespace(search_spans=lambda span_type: [span])


class TestCitationValidity:
    def test_citations_of_retrieved_chunks_pass(self):
        feedback = citation_validity(outputs="Io [3] and Europa [7].", trace=fake_trace([3, 7]))
        assert feedback.value is True

    def test_citation_of_another_chunk_fails_and_names_it(self):
        feedback = citation_validity(outputs="Io [3] and Europa [4].", trace=fake_trace([3, 7]))
        assert feedback.value is False
        assert "[4]" in feedback.rationale


@pytest.mark.parametrize("value, expected", [
    ("yes", 1.0), ("no", 0.0), ("YES", 1.0), (True, 1.0), (False, 0.0),
    (0.5, 0.5), (None, None), (float("nan"), None), ("maybe", None),
])
def test_to_score(value, expected):
    assert to_score(value) == expected


def feedback(name, value):
    return {"assessment_name": name, "feedback": {"value": value}}


def test_means_by_answer_type_counts_only_scored_feedback():
    traces = pd.DataFrame({
        "request": [{"question": "Q1?"}, {"question": "Q2?"}, {"question": "U1?"}],
        "assessments": [
            [feedback("correctness", "yes"), {"assessment_name": "answer_type", "expectation": {}}],
            [feedback("correctness", "no"), feedback("relevance_to_query", None)],  # judge error
            [feedback("refusal", "yes"), feedback("unlisted", "yes")],
        ],
    })
    answer_types = {"Q1?": "answerable", "Q2?": "answerable", "U1?": "unanswerable"}
    names = ["correctness", "relevance_to_query", "refusal"]
    assert means_by_answer_type(traces, answer_types, names) == {
        "correctness/answerable": 0.5,
        "refusal/unanswerable": 1.0,
    }


class TestContextPrecision:
    def relevance_output(self, verdicts):
        """RetrievalRelevance's shape: the '/precision' average first, then one verdict per chunk."""
        average = Feedback(name="retrieval_relevance/precision", value=0.0)
        return [average] + [Feedback(name="retrieval_relevance", value=v) for v in verdicts]

    def test_share_of_chunks_judged_relevant(self):
        feedback = context_precision(self.relevance_output(["yes", "no", "yes", "no", "no"]))
        assert feedback.name == "retrieval_relevance"
        assert feedback.value == 0.4
        assert "2 of 5" in feedback.rationale

    def test_a_failed_chunk_verdict_makes_the_whole_score_an_error(self):
        chunks = self.relevance_output(["yes", "yes"])
        chunks[2] = Feedback(name="retrieval_relevance", error="judge timed out")
        feedback = context_precision(chunks)
        assert feedback.value is None
        assert feedback.error is not None


def test_judge_errors_lists_failed_verdicts_only():
    traces = pd.DataFrame({"assessments": [
        [feedback("correctness", "yes"),
         {"assessment_name": "faithfulness",
          "feedback": {"value": None, "error": {"error_message": "401 invalid x-api-key"}}}],
        [{"assessment_name": "answer_type", "expectation": {"value": "answerable"}}],
    ]})
    assert judge_errors(traces) == ["faithfulness: 401 invalid x-api-key"]
