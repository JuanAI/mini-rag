"""Log a retrieval run to MLflow (--mlflow). Needs requirements-eval.txt."""
EXPERIMENT = "mini-rag-retrieval-eval"


def mlflow_name(metric: str) -> str:
    """The MLflow name for a metric: 'mrr' -> 'overall/mrr',
    'recall@5/jupiter' -> 'by_document/recall_at_5_jupiter'.

    MLflow rejects '@' in names, and its chart view only draws a metric unasked
    when the name has a '<section>/' prefix.
    """
    metric = metric.replace("@", "_at_")
    if "/" in metric:
        return "by_document/" + metric.replace("/", "_")
    return "overall/" + metric


def log_to_mlflow(run_name: str | None, params: dict, metrics: dict[str, float], ranks: list[dict]) -> str:
    """Log one retrieval run to MLflow and return its run id. Plain params and metrics,
    no traces: there is no LLM call to trace."""
    import mlflow

    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)
        mlflow.log_metrics({mlflow_name(name): value for name, value in metrics.items()})
        mlflow.log_dict(ranks, "ranks.json")
    return run.info.run_id
