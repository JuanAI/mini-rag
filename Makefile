# Common commands. Run from the repo root: `make help` lists them.
VENV   ?= .venv
PYTHON := $(VENV)/bin/python
PIP    := $(VENV)/bin/pip

RUN_NAME   ?=
LIMIT      ?= 3
CHUNK_SIZE ?=

OTLP_ENDPOINT := http://localhost:4318
MLFLOW_SERVER := http://localhost:5000
COMPOSE_ALL   := docker compose --profile obs --profile evals --profile langfuse

# The local Langfuse's project keys, created on its first start (see docker-compose.yml).
# Set on the command line, so they win over .env (load_dotenv doesn't override them).
LANGFUSE_LOCAL := LANGFUSE_ENABLED=true LANGFUSE_HOST=http://localhost:3100 \
                  LANGFUSE_PUBLIC_KEY=pk-lf-local-mini-rag LANGFUSE_SECRET_KEY=sk-lf-local-mini-rag

.DEFAULT_GOAL := help
.PHONY: help venv install install-dev install-eval run run-jaeger run-langfuse-local test \
        eval-retrieval eval-retrieval-mlflow update-baseline eval-llm eval-llm-smoke \
        eval-llm-server mlflow-ui obs-up obs-down evals-up evals-down langfuse-up langfuse-down \
        langfuse-reset up down ps logs clean

help: ## List the commands
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

## Setup

venv: ## Create the virtualenv
	test -d $(VENV) || python3 -m venv $(VENV)

install: venv ## Install the bot's dependencies
	$(PIP) install -r requirements.txt

install-dev: venv ## Install the bot plus pytest
	$(PIP) install -r requirements-dev.txt

install-eval: venv ## Install the bot plus MLflow and the Anthropic SDK (LLM eval)
	$(PIP) install -r requirements-eval.txt

## Run

run: ## Start the CLI (Langfuse traces if LANGFUSE_ENABLED=true in .env)
	$(PYTHON) main.py

run-jaeger: ## Start the CLI and send every span to the local Jaeger (make obs-up first)
	OTEL_EXPORTER_OTLP_ENDPOINT=$(OTLP_ENDPOINT) $(PYTHON) main.py

run-langfuse-local: ## Start the CLI with traces to the local Langfuse (make langfuse-up first)
	$(LANGFUSE_LOCAL) $(PYTHON) main.py

## Test and eval

test: ## Run the test suite, including the retrieval gate (no API keys)
	$(PYTHON) -m pytest

eval-retrieval: ## Retrieval report: recall@k, MRR, NDCG@10 (CHUNK_SIZE=120 to re-chunk)
	$(PYTHON) -m eval.retrieval $(if $(CHUNK_SIZE),--chunk-size $(CHUNK_SIZE))

eval-retrieval-mlflow: ## Retrieval report, also logged to MLflow (RUN_NAME=chunk-80, CHUNK_SIZE=...)
	$(PYTHON) -m eval.retrieval --mlflow $(if $(RUN_NAME),--run-name $(RUN_NAME)) $(if $(CHUNK_SIZE),--chunk-size $(CHUNK_SIZE))

update-baseline: ## Rewrite eval/baselines/retrieval.json (commit it deliberately)
	$(PYTHON) -m eval.retrieval --update-baseline

eval-llm: ## LLM judge eval over every question; paid APIs (RUN_NAME=my-change)
	$(PYTHON) -m eval.llm_eval $(if $(RUN_NAME),--run-name $(RUN_NAME))

eval-llm-smoke: ## LLM judge eval on the first LIMIT questions (default 3)
	$(PYTHON) -m eval.llm_eval --limit $(LIMIT)

eval-llm-server: ## LLM judge eval logged to the MLflow container (make evals-up first)
	MLFLOW_TRACKING_URI=$(MLFLOW_SERVER) $(PYTHON) -m eval.llm_eval $(if $(RUN_NAME),--run-name $(RUN_NAME))

mlflow-ui: ## Open the local mlflow.db at http://localhost:5000 (same port as evals-up)
	$(VENV)/bin/mlflow ui

## Local containers (docker compose profiles)

obs-up: ## Start Jaeger: UI http://localhost:16686, OTLP on 4317/4318
	docker compose --profile obs up -d

obs-down: ## Stop Jaeger
	docker compose --profile obs down

evals-up: ## Start the MLflow server at http://localhost:5000
	docker compose --profile evals up -d

evals-down: ## Stop the MLflow server (its volume is kept)
	docker compose --profile evals down

langfuse-up: ## Start the local Langfuse (6 containers, ~3-4 GB RAM): http://localhost:3100
	docker compose --profile langfuse up -d

langfuse-down: ## Stop the local Langfuse (its data is kept)
	docker compose --profile langfuse down

langfuse-reset: ## Stop the local Langfuse and DELETE its traces and project (volumes)
	docker compose --profile langfuse down -v

up: ## Start every local container, including Langfuse
	$(COMPOSE_ALL) up -d

down: ## Stop every local container
	$(COMPOSE_ALL) down

ps: ## Show the local containers and their health
	$(COMPOSE_ALL) ps

logs: ## Follow the local containers' logs
	$(COMPOSE_ALL) logs -f

## Housekeeping

clean: ## Remove caches (keeps .venv, mlflow.db and mlruns/)
	find . -name __pycache__ -type d -not -path './$(VENV)/*' -prune -exec rm -rf {} +
	rm -rf .pytest_cache
