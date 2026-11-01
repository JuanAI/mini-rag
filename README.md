# mini-rag

A small retrieval-augmented generation (RAG) bot built from scratch: no LangChain, no vector database. It reads a folder of `.txt` files, chunks and embeds them locally, retrieves the most relevant passages for a question, and asks an OpenAI model for an answer grounded in, and citing, those passages.

```
Me: What are the four Galilean moons?
RAGBot: The four Galilean moons are Io, Europa, Ganymede, and Callisto [118].

Retrieved sources:
[chunk id 127] data/jupiter.txt
[chunk id 118] data/jupiter.txt
[chunk id 122] data/jupiter.txt
[chunk id 126] data/jupiter.txt
[chunk id 124] data/jupiter.txt

Me: Can you recommend a good restaurant in London?
RAGBot: I don't know based on these documents.

Retrieved sources:
[chunk id 104] data/adalovelace.txt
[chunk id 26] data/adalovelace.txt
[chunk id 79] data/adalovelace.txt
[chunk id 25] data/adalovelace.txt
[chunk id 82] data/adalovelace.txt
```

A real session (`gpt-4o-mini`). `[118]` cites the chunk that names the four moons. Retrieval always returns *k* chunks, even for a question the documents can't answer, so the second answer still lists sources.

## How it works

1. **Parse**: `parser.py` reads each `.txt` file as is. Other file types are skipped. `ParserFactory` is the place to add new formats.
2. **Chunk**: `chunker.py` splits text by *tokens*, not characters, using the embedding model's own tokenizer. Each chunk is 96 tokens: 80 new ones plus 16 repeated from the end of the previous chunk, so context at the boundaries isn't lost. Chunk text is decoded from the tokens, so it is lowercase. Each chunk keeps its source filename.
3. **Embed**: chunks are batch-encoded with [`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) and L2-normalized, so a dot product equals cosine similarity.
4. **Retrieve**: the question is embedded the same way, scored against every chunk with a single matrix product, and the top *k* (default 5) are used.
5. **Generate**: the retrieved chunks go into the prompt, each labelled with its chunk id (`[118] Source: …`). Wikipedia footnote markers like `[155]` are removed from that prompt text only, so the model can't mistake them for passage ids; the indexed chunks keep them, so retrieval is unaffected. The system prompt tells the model to answer only from that context, treat it as data rather than instructions, cite passages by their id (with a real id from the context as the example), and say when the answer isn't there.

A NumPy matrix is enough at this scale. There's no persistence, reranker or conversation memory, by design.

## Project structure

```
mini-rag/
├── data/            # sample corpus: Ada Lovelace, Jupiter, CRISPR (from Wikipedia)
├── parser.py        # file → Document
├── chunker.py       # token-based overlapping chunker
├── rag.py           # RAGBot: index, retrieve, ask
├── telemetry.py     # OpenTelemetry tracing, Langfuse export
├── models/          # bundled all-MiniLM-L6-v2 embedding model (~87 MB)
├── main.py          # interactive CLI
├── docker-compose.yml  # optional Jaeger, MLflow and Langfuse, opt-in profiles
├── Makefile         # shortcuts for the commands below (`make help`)
├── .vscode/         # run configuration and test settings for VS Code
├── .github/workflows/  # CI (lint and tests on every push) and the manual LLM eval
├── pyproject.toml   # dependencies and extras, pytest and ruff settings; uv.lock pins the versions
├── eval/            # eval dataset; retrieval/ (metrics, MLflow and Langfuse logging); llm/ (MLflow LLM judges)
├── docs/            # PLAN.md (build steps) and TDD.md (contracts)
└── tests/           # pytest suite (real embedding model, mocked OpenAI); eval/ holds the eval tests
```

## Setup

Requires Python 3.11+ (Langfuse 5 needs it) and [uv](https://docs.astral.sh/uv/). CI tests 3.11, 3.12 and 3.13.

```bash
uv sync                # creates .venv with the versions pinned in uv.lock
source .venv/bin/activate
cp .env.example .env   # then add your OpenAI or Azure OpenAI settings
```

Dependencies are in `pyproject.toml`, with two optional extras: `dev` (pytest, ruff) and `eval` (MLflow and the Anthropic SDK, for the LLM eval). `uv sync --extra dev --extra eval` installs everything. `uv sync` installs exactly the set you ask for, so leaving out an extra removes it. On Linux, torch comes from PyTorch's CPU index, so the CUDA build (several GB) isn't downloaded.

`.env` variables. Fill in **one** provider:

| Variable | Provider | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | OpenAI | API key |
| `AZURE_OPENAI_ENDPOINT` | Azure | Resource endpoint; when it is set, Azure is used |
| `AZURE_OPENAI_API_KEY` | Azure | API key |
| `OPENAI_API_VERSION` | Azure | API version, e.g. `2024-10-21` |
| `MODEL_NAME` | both | Chat model (default `gpt-4.1-mini`); for Azure, the deployment name |

Or use the Makefile: `make install-dev` (or `make install-eval` for everything), then `make help` lists every command (run, test, evals, local containers).

The embedding model is bundled in `models/all-MiniLM-L6-v2`, so nothing is downloaded at runtime. Run commands from the repo root, because the model path is relative.

## Usage

From the repo root:

```bash
python main.py
```

Or, in VS Code, open the folder and press **F5**. The "RAGBot: main.py" configuration uses `.venv` and the integrated terminal. Ctrl-D or Ctrl-C ends the session.

Ask about anything in `data/`. To use your own documents, drop `.txt` files into `data/`.

You can also call the bot from Python:

```python
from rag import RAGBot

bot = RAGBot()
bot.read_and_embed_data("data/")
print(bot.ask("What does CRISPR stand for?", k=3))
```

## Tracing

The bot emits OpenTelemetry spans for every question. Tracing is off unless configured, and a tracing failure never breaks an answer.

| Span | Holds |
|---|---|
| `rag.ask` (root) | question, answer, `session.id`, `user.id`, k |
| `rag.retrieve` | retrieved chunk ids, sources, top score |
| `rag.generate` | model, input and output tokens |
| `openai.chat` | prompt, completion, model and token usage, from the OpenAI instrumentor |

The code only talks to OpenTelemetry. One `TracerProvider` (`telemetry.py`) hands every finished span to each attached processor, and each one exports on its own, so Jaeger never forwards to Langfuse or the reverse:

```
 rag.ask ─► rag.retrieve / rag.generate ─► openai.chat      (each step is a span)
                        │
 ┌──────────────────────┼──────── mini-rag process ─────────────────────────┐
 │                      ▼                                                   │
 │        OpenTelemetry SDK: TracerProvider (setup_telemetry)               │
 │                      │                                                   │
 │          ┌───────────┴──────────────────────┐                            │
 │          ▼                                  ▼                            │
 │  OTLP exporter                      Langfuse span processor              │
 │  if OTEL_EXPORTER_OTLP_ENDPOINT     if LANGFUSE_ENABLED=true             │
 │  is set; all spans                  (setup_langfuse); HTTP spans dropped │
 └──────────┼──────────────────────────────────┼────────────────────────────┘
            │ OTLP/HTTP                        │ OTLP/HTTP + API keys
            ▼                                  ▼
  ┌────────────────────┐            ┌────────────────────────────┐
  │ Jaeger :4318       │            │ Langfuse :3100 (local)     │
  │ UI :16686          │            │ or cloud.langfuse.com      │
  │ timing and errors  │            │ prompts, tokens, cost,     │
  │ (an APM in prod)   │            │ sessions, users, scores    │
  └────────────────────┘            └────────────────────────────┘
```

Either path, both or neither can be on. With neither, spans are created and dropped at almost no cost. A production setup often puts an OpenTelemetry Collector between the app and the backends to do this fan-out; mini-rag does it in-process.

To send traces to [Langfuse](https://langfuse.com) (the free cloud tier is enough), set these in `.env`:

```bash
LANGFUSE_ENABLED=true
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com   # must match the keys' region (EU here; US is us.cloud.langfuse.com)
```

Or run Langfuse locally, with no account: `make langfuse-up` starts a self-hosted copy (six containers, about 3–4 GB of RAM; the first start takes a minute or two) and creates a `mini-rag` project with API keys. Then `make run-langfuse-local` starts the bot with those keys, without editing `.env`, and the traces appear at http://localhost:3100 (the login is in the comments of `docker-compose.yml`). `make langfuse-down` stops it and keeps the data; `make langfuse-reset` deletes it.

Each `python main.py` run is one Langfuse session, so **Sessions** shows a conversation's questions together. `user.id` is `RAGBOT_USER_ID`, or your OS user name. If HTTP-client instrumentation is ever added, its spans are kept out of Langfuse by a filter.

To see **every** span locally, as well as or instead of Langfuse, run Jaeger and point the OTLP exporter at it:

```bash
docker compose --profile obs up -d
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318 python main.py   # then open http://localhost:16686
```

Langfuse is for the live bot and MLflow for offline evals: `eval/llm/` never sets up tracing, so eval runs don't reach Langfuse. To log evals to an MLflow server instead of the local `mlflow.db`, run `docker compose --profile evals up -d` and set `MLFLOW_TRACKING_URI=http://localhost:5000`.

## Tests

```bash
uv sync --extra dev
pytest
ruff check .      # or: make test, make lint
```

The tests use the real embedding model and a mocked OpenAI client, so they make no paid API calls.

GitHub Actions runs the same on every push and pull request (`.github/workflows/ci.yml`): `uv sync --locked --extra dev`, then `ruff check` once and `pytest` on Python 3.11, 3.12 and 3.13. `--locked` fails the build if `uv.lock` is out of date with `pyproject.toml`. It needs no secrets, because the embedding model is committed and OpenAI is mocked. `HF_HUB_OFFLINE=1` makes sure nothing is downloaded.

## Evaluation

| What | Tool | Where it runs |
|---|---|---|
| Retrieval quality: Recall@k, MRR and NDCG@10 over 50 questions, each with a keyword that marks its answer passage | plain Python (`python -m eval.retrieval`); add `--mlflow` or `--langfuse` to log the run for comparison in `mlflow ui` or Langfuse | `pytest` (`tests/eval/retrieval/test_metrics.py`). Fails if recall@5 or MRR drops below `eval/baselines/retrieval.json` |
| Citation validity: every `[n]` in an answer is a retrieved chunk id | plain Python (`eval/checks.py`) | Unit-tested by `pytest`; scored on real answers by the LLM eval |
| Answer groundedness, relevance, correctness, refusal on out-of-scope questions | [MLflow GenAI](https://mlflow.org/docs/latest/genai/) LLM judges | On demand: `python -m eval.llm`, then `mlflow ui` |

CI runs `pytest`, so the retrieval gate and the citation checks run on every push and PR. The LLM eval doesn't: it costs money and its scores vary.

Current retrieval baseline: **recall@5 0.90, MRR 0.688** (top 1: 0.52, top 3: 0.84, top 10: 0.96; NDCG@10 0.643).

To compare retrieval settings over time, log each run to MLflow (needs the `eval` extra, no API keys). Runs go to the `mini-rag-retrieval-eval` experiment, with the settings as params and each question's rank in `ranks.json`. The scores are deterministic, so log a run only when a setting changes:

```bash
python -m eval.retrieval --mlflow --run-name chunk-80
python -m eval.retrieval --mlflow --chunk-size 120 --run-name chunk-120
mlflow ui                                      # charts under overall/ (recall_at_k, mrr, ndcg_at_10) and by_document/
```

Or compare them in Langfuse: `--langfuse` publishes the 50 questions as the dataset `mini-rag-retrieval` and records an experiment run on it, with each question's `rank` and `hit@5` and the run's recall@k, MRR and NDCG@10. It uses the `LANGFUSE_*` keys and host, and re-runs update the same dataset items. With the local Langfuse (`make langfuse-up`):

```bash
make eval-retrieval-langfuse-local RUN_NAME=chunk-80
make eval-retrieval-langfuse-local RUN_NAME=chunk-120 CHUNK_SIZE=120
# then Datasets → mini-rag-retrieval → Runs, select both runs to compare them question by question
```

If Langfuse is unreachable or a key is missing, the report still prints and the command exits 0, with a warning.

`pytest` needs no API keys. The LLM eval calls paid APIs and its scores vary from run to run, so it is run by hand whenever the prompt or model changes:

```bash
uv sync --extra dev --extra eval
# set ANTHROPIC_API_KEY in .env (see .env.example): the judge is Claude Haiku 5.5, a different model family from the bot
python -m eval.llm --limit 3              # smoke test
python -m eval.llm --run-name my-change   # every question
mlflow ui                                      # compare runs
```

Each run logs the mean of every judge, overall and per question type (answerable or unanswerable).

It can also run on GitHub: **Actions → LLM eval → Run workflow** (`.github/workflows/llm-eval.yml`), with the question limit, run name and generator model as inputs. It needs the repo secrets `OPENAI_API_KEY` and `ANTHROPIC_API_KEY`, and uploads the MLflow store (`mlflow.db`, `mlruns/`) as an artifact: download it and run `mlflow ui` next to it.

Current LLM eval results: MLflow run **`citation-fix`** (generator `gpt-4o-mini`, judge Claude Haiku 5.5, k=5, 55 questions, 0 judge errors), with the earlier run **`baseline`** for comparison:

| Answerable (50) | `citation-fix` | `baseline` |
|---|---|---|
| correctness | 0.86 | 0.88 |
| relevance_to_query | 0.94 | 0.96 |
| faithfulness | 0.92 | 0.82 |
| retrieval_sufficiency (context recall) | 0.86 | 0.83 |
| retrieval_relevance (context precision) | 0.36 | 0.40 |
| **citation_validity** | **1.00** | **0.68** |

On the 5 unanswerable questions, refusal and citation_validity are 1.00 in both runs.

Expect the judge to vary from run to run, so compare runs by differences bigger than that. With the bot unchanged, faithfulness has scored 0.88 and 0.82. Reviewing the second run showed about 3 of 50 verdicts where the judge's "no" contradicts its own rationale. Context precision is low by design: each question has one answer passage, and the bot retrieves 5 chunks.

In `baseline`, 15 of 50 answers cited passages by position (`[1]`, `[2]`) instead of by chunk id, because the prompt said "Cite supporting passages with [1], [2] etc.", and a few copied Wikipedia footnote markers (e.g. `[155]`). `citation-fix` changed the prompt to cite by id and removed the markers from the prompt text, and every citation now names a retrieved chunk. Retrieval didn't change, so the context precision difference is judge noise; the faithfulness gain is bigger than the usual ±0.06 but comes from one run.

## Limitations

- Top-k retrieval always returns *something*. Similarity scores are not calibrated confidence.
- Grounding instructions reduce unsupported answers but don't guarantee correctness.
- The whole index lives in memory and is rebuilt on every start.

## Data

The sample texts in `data/` are excerpts from Wikipedia articles, licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
