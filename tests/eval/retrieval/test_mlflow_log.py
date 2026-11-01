"""eval/retrieval/mlflow_log.py: metric names and the logged run (skipped without MLflow)."""
import pytest

from eval.retrieval.mlflow_log import log_to_mlflow, mlflow_name


class TestMlflow:
    @pytest.mark.parametrize("metric, expected", [
        ("mrr", "overall/mrr"),
        ("recall@5", "overall/recall_at_5"),
        ("ndcg@10", "overall/ndcg_at_10"),
        ("recall@5/jupiter", "by_document/recall_at_5_jupiter"),
    ])
    def test_name_has_a_chart_section(self, metric, expected):
        assert mlflow_name(metric) == expected

    def test_logs_params_metrics_and_ranks(self, tmp_path, monkeypatch):
        mlflow = pytest.importorskip("mlflow")
        monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
        monkeypatch.chdir(tmp_path)  # artifacts go to ./mlruns, so keep them out of the repo
        ranks = [{"id": "q01", "source": "adalovelace", "question": "q", "rank": 2}]

        run_id = log_to_mlflow("test", {"k": 5}, {"recall@5": 0.9, "mrr": 0.5}, ranks)

        run = mlflow.get_run(run_id)
        assert run.data.params == {"k": "5"}
        assert run.data.metrics == {"overall/recall_at_5": 0.9, "overall/mrr": 0.5}
        assert mlflow.artifacts.load_dict(f"{run.info.artifact_uri}/ranks.json") == ranks
