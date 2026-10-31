# PLAN — implementation

How mini-rag gets built, step by step, test-first, against the contracts in [TDD.md](TDD.md).

Each step follows the same loop:
1. **Red**: write the tests for that step's contracts and watch them fail.
2. **Green**: write the smallest change that makes them pass.
3. **Refactor**: clean up while the tests stay green.
4. Commit.

Steps 1–5 run without API keys; Step 3 tests mock OpenAI. Only Step 6 calls paid APIs. Step 7's tests need no network; its manual check needs Langfuse keys (the free tier is enough).

---

## Step 0 — Environment and baseline ✅

- [x] `python -m venv .venv && pip install -r requirements-dev.txt`
- [x] `cp .env.example .env`. The key stays empty for now; retrieval and tests don't need it.
- [x] Set up the test layout: `tests/` holds the tests, and `pyproject.toml` sets `pythonpath` and `testpaths`. Shared fixtures go into `tests/conftest.py` in the step that first needs them (Steps 2 and 3).
- [x] Check that `pytest` runs. It collects no tests yet: Steps 1–4 add each module together with its tests.

No application code is committed in this step. The earlier prototype is kept only as local reference; each module is rebuilt test-first in its own step.

**Done when**: the dependencies install cleanly and `pytest` runs. ✅

## Step 1 — Parsing (`parser.py`) ✅

Tests first:
- [x] `TxtParser.parse_file`: returns one page, `page_number == 1`, `file_type == "txt"`, and the text exactly as it is in the file.
- [x] `ParserFactory.supports`: `.txt` and `.TXT` → `True`; `.md` and files with no extension → `False`.
- [x] `ParserFactory.get_parser`: raises `NotImplementedError` for unsupported files.

Implementation and cleanup:
- [x] Rename `txtParser` to `TxtParser`.
- [x] Remove the fields nobody uses from `Page` (`chunks`, `embeddings`) and the `numpy` import.
- [x] Drop the old `lint_text` cleanup. Measured on the three data files, the token IDs are identical with and without it, because the tokenizer already does the same cleanup.

**Done when**: all parser tests pass with no model loaded. ✅ 11 passed in 0.01 s.

**Refactor (after Step 1):** `parsing.py` was replaced with the parser from an earlier version of this project:
- `Document` is just `text` and `type`, and `txtParser.parse()` returns it.
- `ParserFactory.get_parser()` returns `None` with a warning for unsupported files, instead of raising, and there is no `supports()`.
- `Chunk` moves to `chunker.py`.
- `tests/test_parser.py` was rewritten to match: 9 tests.

## Step 2 — Chunking (`chunker.py`) ✅

`chunker.py` is the chunker from the earlier version of this project, copied exactly.

Tests (`tests/test_chunker.py`, 11):
- [x] Defaults: `chunk_size == 80`, `chunk_overlap == 0.2`.
- [x] Each window holds `chunk_size + overlap` tokens and starts `chunk_size` tokens after the previous one. Size 5 → `[0..5] [5..10] [10..11]`; 200 tokens with the defaults → windows of 96, 96 and 40 starting at 0, 80 and 160.
- [x] Consecutive chunks share exactly `overlap` (16) tokens.
- [x] Text that fits gives one chunk; blank text gives `[]`.
- [x] `id` counts from 0 within each document and restarts for the next one; `source == str(path)`.
- [x] With the real model:
  - `Tokeniser` adds no special tokens;
  - decoding lowercases the text;
  - every chunk of `data/*.txt` fits the model's limit (at most 99 of 254 tokens).
- [x] `tests/conftest.py` provides a session-scoped `model` fixture.

Known behaviour, kept as is:
- Chunk text is **decoded** from tokens, so it comes out lowercase and without line breaks.
- `id`s restart at 0 for each file, so they aren't unique across the index. `rag.py` cites `[chunk.id]` plus the source, so a citation needs both.
- Size and overlap are attributes (`chunker.chunk_size = N`), not constructor arguments.

**Done when**: the chunker tests pass. ✅ 20 passed in total (9 parser + 11 chunker) in ~4 s. The data files give 111, 114 and 147 chunks.

## Step 3 — RAG engine (`rag.py`) ✅

Brings in `rag.py` from the earlier version of this project. Indexing, retrieval and answer generation are covered in this one step.

Changes from the earlier version:
- [x] `parsing.py` renamed back to `parser.py`, the name `rag.py` imports.
- [x] The embedding model is bundled in `models/all-MiniLM-L6-v2` (87 MB), so the relative path in `rag.py` works on a fresh clone. `models/` is no longer ignored by git.
- [x] Both providers are supported: Azure OpenAI when `AZURE_OPENAI_ENDPOINT` is set, otherwise OpenAI. The model comes from `MODEL_NAME`, default `gpt-4.1-mini`.
- [x] `httpx.Client(verify=False)` removed, so HTTPS certificates are checked again.
- [x] The OpenAI client is still created in `__init__` (kept as is), so a key must be set even to index.
- [x] Unused `httpx` and `pandas` imports removed, so neither is needed in `requirements.txt`.
- [x] Prompt typos fixed: missing spaces between sentences, "dont't" → "don't", "Question" → "Question: ", and a full stop after "Be concise".

Tests (`tests/test_rag.py`, 15, with OpenAI mocked):
- [x] **Client choice:**
  - OpenAI without an Azure endpoint, Azure with one, each with the right settings;
  - the bundled model path is used.
- [x] **Indexing:**
  - one unit-length vector per chunk;
  - all three documents are indexed;
  - `.DS_Store` and `.md` are skipped.
- [x] **Retrieval:**
  - the Ada Lovelace, Jupiter and CRISPR questions put a chunk from the right document first;
  - `k` chunks are sent;
  - with fewer chunks than `k`, all of them are sent.
- [x] **Generation:**
  - model name and `temperature=0`;
  - the exact corrected system prompt;
  - the user message format;
  - `MODEL_NAME` defaults to `gpt-4.1-mini`;
  - the reply lists the sources;
  - `OpenAIError` is re-raised with context.
- [x] `tests/conftest.py` loads the bundled model and provides a session-scoped `bot` with a dummy key.

Manual check with a real key:
- [ ] A question for each topic, plus one outside the documents (e.g. a restaurant recommendation), which should get "don't know".

**Done when**: the mocked tests pass offline (✅ 35 passed in ~5 s: 9 parser + 11 chunker + 15 rag), and the manual check answers correctly with correct citations.

## Step 4 — CLI (`main.py`) ✅

`main.py` comes from the earlier version of this project, with two additions:
- Ctrl-D or Ctrl-C ends the session cleanly instead of printing a traceback.
- A failed request (e.g. an expired API key) is reported and the session continues.

 It runs from the repo root, because `data/` and `models/` are relative paths.

Tests first (`tests/test_cli.py`, 5):
- [x] With a fake bot, each question gets `RAGBot: {answer}` and the greeting is printed.
- [x] Ctrl-D (`EOFError`) and Ctrl-C (`KeyboardInterrupt`) end the loop without an exception. Both failed against the exact copy.
- [x] A real run from the repo root with empty stdin exits with status 0 and no traceback.
- [x] When `ask()` raises, `RAGBot: Sorry, that request failed: …` is printed and the next question is still answered. Found in a real run with an expired key.

VS Code setup (`.vscode/`, from the earlier version of this project):
- [x] `launch.json`: a "RAGBot: main.py" run configuration that uses `.venv`, the integrated terminal (for the input loop) and the repo root as working directory.
- [x] `settings.json`: `.venv` as the interpreter, and the Test Explorer running pytest on `tests/`.
- [x] Comments give the real setup commands (`python -m venv .venv && .venv/bin/pip install -r requirements-dev.txt`), not `make venv`.

**Done when**: `python main.py` works from the repo root, and the VS Code run button starts the bot. ✅ 40 passed in ~7 s (9 parser + 11 chunker + 15 rag + 5 cli).

## Step 5 — Retrieval eval (pytest gate) ✅

Adapted from the earlier `eval_recall.py`. No retrieval-eval library is needed, and `requirements-dev.txt` already has `pytest`.

- [x] `eval/dataset.jsonl`:
  - the 50 questions from `eval_recall.py` (20 Ada Lovelace, 14 CRISPR, 16 Jupiter);
  - 5 unanswerable questions: a restaurant, Bitcoin, the World Cup, Saturn's moons (close to the corpus, but not in it) and the boiling point of ethanol.
  - `expected_answer` comes in Step 6.
- [x] `eval/retrieval.py` and `eval/checks.py`, run with `python -m eval.retrieval [--chunk-size N] [--k N] [--update-baseline]`. A `--show` flag was parsed but never implemented, and was removed in Step 6: the per-question rank table already lists the misses.

Changes from the original contract:
- [x] The gate is on **recall@5** rather than recall@3, because `ask()` now sends the top 5 chunks to the LLM.
- [x] `--chunk-size` overrides `Chunker.chunk_size` while indexing, as `eval_recall.py` did. `RAGBot` has no `chunker` attribute.
- [x] `invalid_citations(answer, allowed)` checks against the retrieved chunk ids, because that's how `rag.py` labels passages.

Tests (`tests/eval/test_eval_retrieval.py`, 17):
- [x] `norm`: `£3,000` → `3 000`; `MIL-STD-1815` → `mil std 1815`; and two more.
- [x] **Dataset:**
  - ids are unique, and the 50 answerable questions have a keyword and a source;
  - every keyword exists in its source document (`invalid_keywords == []`).
- [x] `rank_of_answer`:
  - with a fake index, a better-scoring chunk from the wrong document is skipped;
  - `None` when the keyword is only in another document.
- [x] `evaluate_retrieval`:
  - every metric is in `[0, 1]`;
  - recall never decreases as k grows;
  - per-document recall is reported;
  - a second run gives identical numbers.
- [x] `invalid_citations`: valid ids, an unknown id, `[0]`, no citations, repeats, and non-numeric brackets.
- [x] **Gate**: `recall@5` and `mrr` are each ≥ baseline − 0.02.

**Baseline** (`eval/baselines/retrieval.json`, chunk size 80):

| recall@1 | recall@3 | recall@5 | recall@10 | MRR |
|---|---|---|---|---|
| 0.52 | 0.84 | **0.90** | 0.96 | **0.688** |

Recall@5 by document: Ada Lovelace 0.80, CRISPR 1.00, Jupiter 0.94.

Misses at k=5: q06 (Menabrea's article, rank 10), q07 (seven notes, rank 86), q11 (date of meeting Babbage, rank 9), q12 ("Enchantress of Number", rank 15), q36 (Jupiter's diameter, rank 9).

**Chunk size comparison** (`--chunk-size`):

| Size | Chunks | recall@5 | MRR | Note |
|---|---|---|---|---|
| **80** (default) | 372 | **0.90** | 0.688 | |
| 120 | 248 | 0.88 | 0.671 | |
| 200 | 150 | 0.80 | 0.644 | |
| 254 | 118 | 0.90 | 0.725 | **Not valid**: 254 + 20% overlap = 304 tokens, over the model's 256 limit, so the end of each chunk is silently cut off when embedding |

The default of 80 is the best valid size. Any size above about 211 overflows the model, because each chunk is the size plus 20%.

Open issue found while building the citation check:
- [ ] `rag.py` labels passages with chunk ids (`[37] Source: …`), but its prompt says *"Cite supporting passages with [1], [2] etc."* The model may cite numbers that match no passage. Fix the prompt or the labels before Step 6.

**Done when**: `pytest` runs the retrieval gate in seconds, with no API keys. ✅ 57 passed in ~6 s (40 before + 17 eval).

## Step 6 — LLM answer eval (on demand) ✅

- [x] `requirements-eval.txt`: `mlflow>=3.17` and `anthropic>=2.3`. `mlruns/` and `mlflow.db` (MLflow's default local store) added to `.gitignore`.
- [x] `expected_answer` for the 50 answerable questions, in the documents' own wording.
- [x] `RAGBot.ask` split into `retrieve` and `generate`, so the eval can trace retrieval and generation as separate spans. `ask`'s output is unchanged.
- [x] `eval/llm_eval.py` with the scorers listed in TDD.md, plus offline tests for its helpers (`tests/eval/test_llm_eval.py`, skipped when MLflow isn't installed).
- [x] Smoke-tested with `--limit 3`, and on one answerable and one unanswerable question. There was no Anthropic key yet, so these runs used `JUDGE_MODEL=openai:/gpt-4.1-mini` to check the wiring only, and were then deleted. What they caught:
  - with OpenAI autologging on, retrieval ended up in a separate trace and the retrieval judges found no context, so `predict` is now traced explicitly;
  - groundedness and relevance scored correct refusals "no", so every judge now scores a single `answer_type`.
- [x] Full run with the Claude judge (Haiku 5.5): 55 questions, 0 judge errors. Every failing verdict checked by hand (about 40):
  - **Real bot failures, judged correctly**:
    - q07 and q12 answered "don't know" (retrieval misses: rank 86 and rank 15);
    - q11 gave no day, because the retrieved chunk has none;
    - q36 gave the equatorial-minus-polar difference (9,276 km) as the diameter;
    - q39 said 1878 instead of 1831;
    - q06 doubled "Federico" in Menabrea's name.
  - **Judge too strict**: q05 sufficiency and q16 groundedness.
  - **Label errors, fixed in the dataset**: q14 expected "Lord Byron", a name the retrieved text doesn't give; q37 said "about" ten hours where the document says "slightly less than".
  - `refusal` was right on all 5 unanswerable questions, so its instructions were not tightened.
- [x] Baseline recorded in the README: run `baseline`.
- [x] Follow-ups after walking through the charts:
  - `retrieval_groundedness` renamed `faithfulness`;
  - `retrieval_relevance` (context precision) added;
  - MLflow's duplicate `/mean` metrics dropped;
  - judge errors now fail the run, so missing scores aren't mistaken for real ones. On its first run, this check caught a bug in the renaming wrappers.

  All earlier runs were deleted and purged, and `baseline` was run again. Faithfulness came in at 0.82 against 0.88 before, with the bot unchanged. About 3 of 50 Haiku verdicts contradict their own rationale, so expect run-to-run noise of about ±0.06 on that metric.

- [x] Retrieval eval in MLflow too: `python -m eval.retrieval --mlflow [--run-name NAME]` logs params, metrics and a `ranks.json` artifact to the `mini-rag-retrieval-eval` experiment. NDCG@10 added to its metrics (0.643 at chunk size 80). The baseline file stays the pytest gate.

**Done when**: one command produces a run in `mlflow ui` with overall scores and scores per `answer_type`. ✅

Found by this step, for a later fix:
- [ ] **Citations**: 15 of 50 answerable answers cite passages by position (`[1]`, `[2]`) rather than by chunk id, which the system prompt's "Cite supporting passages with [1], [2] etc." invites. A few cite Wikipedia footnote markers (e.g. `[155]`), which `txtParser` leaves in the text. Possible fixes: give an id example from the context in the prompt, and strip `[n]` footnotes when parsing.

## Step 7 — Production tracing with Langfuse

Goal: run mini-rag the way a production service is observed, modelled on a production enterprise-level RAG app. Live requests are traced with OpenTelemetry, and the LLM-relevant spans go to Langfuse, grouped by session and user. MLflow stays the offline eval tool. Contracts: TDD.md → Observability.

Why this split: in a survey of about 27 open-source projects that integrate Langfuse (app builders, RAG apps, agents, frameworks; October 2026), this is the common pattern:
- **Langfuse is used to watch production, not to test.** Nearly all use it for tracing only, a few also record user feedback as scores, and none of the repos uses its datasets or experiments for pre-release testing. It is usually one tracing provider among several.
- **Offline eval is done with something else.** That means the framework's own evaluators, RAGAS, a hand-written script, or a separate platform (Onyx: Langfuse for tracing, Braintrust for evals).
- **Evals come in two layers:** cheap deterministic checks (retrieval hit@k or rank, expected tool calls) and LLM-judge checks (faithfulness, correctness) run by hand when something changes.
- **LLM quality rarely blocks a release.** CI runs ordinary tests; LLM evals run by hand, on a schedule, or only when a prompt changes, and report rather than block.
- **Mature teams feed production back into testing** (AutoGPT, LibreChat): real questions or tool calls are exported from traces, labelled, and frozen into the test set.

mini-rag already has the two layers (`eval/retrieval.py` as a pytest gate, which CI will run from Step 8; `eval/llm_eval.py` by hand). This step adds the production side, plus the feedback loop below.

- [x] Add `opentelemetry-api`, `opentelemetry-sdk`, `opentelemetry-exporter-otlp-proto-http` (only OTLP/HTTP is used, so no gRPC), `opentelemetry-instrumentation-openai` and `langfuse` to `requirements.txt`, at versions checked against the installed SDKs (Langfuse 5.1, OTel 1.45). The Langfuse API details (span processor, filter hook, scores, datasets) are taken from the installed SDK, not assumed.
- [ ] Langfuse project: Langfuse Cloud, free tier, in the region the keys are for. Alternatively, self-host with Langfuse's Docker Compose.
- [x] Add the `LANGFUSE_*` placeholders to `.env.example`.

Tests first (`tests/test_telemetry.py`; spans captured with OTel's in-memory exporter, no network):
- [x] `setup_langfuse()` returns `False` and attaches nothing when `LANGFUSE_ENABLED` is unset or `false`.
- [x] With `LANGFUSE_ENABLED=true` and a key missing, it returns `False`, logs a warning, and doesn't raise.
- [x] `should_export_to_langfuse`: `httpx`, `requests`, `urllib` and `aiohttp` instrumentation scopes → `False`; the `mini_rag` and OpenAI instrumentor scopes → `True`.
- [x] `setup_telemetry()` is idempotent and sets `service.name`.
- [x] `ask()` (mocked OpenAI client) emits one trace, with root `rag.ask` and children `rag.retrieve` and `rag.generate`.
- [x] `rag.ask` carries `langfuse.observation.input`, `langfuse.observation.output`, `session.id` and `user.id`. `rag.retrieve` carries the retrieved chunk ids; `rag.generate` carries the model and token counts, as `rag.*` attributes: `gen_ai.*` names would make Langfuse cost the call a second time next to the OpenAI instrumentor's span.
- [x] On `OpenAIError`, the spans have status `ERROR` with the exception recorded, and `ask()` still raises as before.
- [x] `ask()` without ids, and with no provider set up, behaves exactly as before. `test_rag.py` passes unchanged. `test_cli.py`'s fake bot had to accept the two new keyword arguments.
- [x] `main.py` passes the same `session_id` to every `ask()` in one run, and flushes on exit (Ctrl-D, Ctrl-C).
- [x] `session.id` and `user.id` are on every span of the trace (Langfuse's `propagate_attributes`), not only `rag.ask`: Langfuse v4 sums a session's or user's cost over the spans that carry the id, and with the ids on the root only the Sessions and Users pages showed no cost.

Then:
- [x] Implement `telemetry.py`, the spans in `rag.py`, and the session and flush in `main.py`. ✅ 111 passed (24 new). Checked with a real OpenAI client over a fake HTTP transport: the instrumentor's `openai.chat` span sits under `rag.generate` and passes the Langfuse filter.
- [ ] Manual check: with `LANGFUSE_ENABLED=true`, ask three questions in one CLI session. In Langfuse, **Sessions** should show one session with three traces, and each trace should show `retrieve` and `generate` with chunk ids, model, tokens and cost. Then ask one question with the network off: the bot still answers, and a warning is logged.
- [x] `docker-compose.yml` with two opt-in profiles, `evals` and `obs` (TDD.md → Local services):
  - `evals`: an MLflow server (official image, same version as the installed `mlflow` client) on port 5000, with a named volume.
  - `obs`: Jaeger, with its UI on 16686 and OTLP on 4317/4318.

  Nothing starts by default, and Langfuse isn't in it: mini-rag uses Langfuse Cloud. Images pinned: `ghcr.io/mlflow/mlflow:v3.17.2`, `jaegertracing/jaeger:2.22.0`. Checked with `docker compose config`; not started yet.
- [x] A `langfuse` compose profile for running Langfuse offline: Langfuse's official v4 compose file, pinned (4.63.0), localhost-only ports, telemetry off, and a `mini-rag` project with API keys created on first start. Makefile: `langfuse-up`, `langfuse-down`, `langfuse-reset`, `run-langfuse-local`. Checked with `docker compose config`; not started yet.
- [x] Check `docker compose --profile langfuse up -d` and `make run-langfuse-local`. ✅ Langfuse 4.63.0 healthy in about 10 s after the databases; the auto-created keys pass `auth_check()`. Three CLI questions gave 3 traces in **one** session, 4 observations each (`rag.ask` root with session and user, `rag.retrieve`, `rag.generate`, `openai.chat`). Only `openai.chat` is a costed GENERATION (`gpt-4o-mini`, 690 tokens, $0.00011), so cost isn't counted twice. Host ports moved to 3100/9190 after a clash with another local Langfuse on 3000/9090.

  Facts for the Langfuse-reading tasks below: v4 runs in "events_only" mode, where `GET /api/public/traces` and `/observations` return an error; reads go through `/api/public/v2/observations` (`lf.api.observations.get_many(name=..., session_id=..., fields=..., expand_metadata="attributes")`), whose input and output come back as raw strings. Span attributes land in metadata as `attributes.<name>`, e.g. `attributes.rag.chunk_ids`.
- [ ] Check `docker compose --profile obs up -d`: with `OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318`, a CLI question shows up in Jaeger with every span. mini-rag doesn't instrument `httpx`, so there are no HTTP spans for the Langfuse filter to drop; the filter guards against an HTTP instrumentor being added later. This stands in for a production APM pipeline.
- [ ] Check `docker compose --profile evals up -d`: `MLFLOW_TRACKING_URI=http://localhost:5000 python -m eval.llm_eval --limit 3` logs to the container, and the run appears at http://localhost:5000. Without the variable, the eval still uses the local `mlflow.db`.
- [ ] Optional, a deterministic eval in Langfuse: `python -m eval.retrieval --langfuse` publishes a dataset plus an experiment run, with per-item `rank` and `hit@5` and run-level recall@k and MRR. Run it at chunk sizes 80 and 120 and compare the two runs in Langfuse → Datasets → Runs.
- [ ] Production → test set: `python -m eval.from_langfuse [--session ID | --since DATE]` reads `rag.ask` traces from Langfuse and prints candidate rows in the `eval/dataset.jsonl` format: the question, the bot's answer and retrieved chunk ids, with `id`, `answer_type`, `keyword`, `source` and `expected_answer` left blank. It never appends by itself: a person picks the interesting ones (wrong answers, refusals, thumbs-down) and labels them, and the retrieval and LLM evals then cover them from the next run. Tested offline: a trace-to-row helper over a stored trace fixture, no network.
- [x] README: a "Tracing" section with setup, what each span holds, and the Langfuse/MLflow split.

Not in this step: scoring live production traces with LLM judges. Online grading would be its own follow-up.

**Done when**: a CLI session appears in Langfuse as one session of complete traces; turning Langfuse off or breaking it never affects answers; and `pytest` passes with no network.

## Step 8 — Polish and publish

- [ ] Search the repo for secrets: no `sk-` strings, `.env` is ignored by git, and `.env.example` contains only placeholders.
- [ ] Check that README examples match real output. Replace the illustrative sample session with a real one.
- [ ] Add a GitHub Actions workflow (`ci.yml`) that, on every push and PR, runs `pytest` on Python 3.10–3.12, covering the unit tests, the retrieval gate and the citation checks. Cache the Hugging Face model between runs. It needs no secrets.
- [ ] Optional: a `llm-eval.yml` workflow triggered manually (`workflow_dispatch`), using API keys stored as repo secrets.
- [ ] Optional: a `ruff` config.
- [x] `git init`, make the first commit, add the remote `git@github.com:JuanAI/mini-rag.git`, and push to `main`. Done early: each step since Step 4 has been merged by PR.

**Done when**: CI is green on `main`, and the repo on GitHub matches the README.

---

## Out of scope for now

Persistence for the vector index, a reranker, more file types (PDF, Markdown), conversation memory, streaming responses, and LLM-judge scoring of live traces in Langfuse. Each would be its own follow-up, with contracts added to TDD.md first. Step 7's sessions group traces only; the bot still answers each question without memory of the earlier ones.
