# TDD — signatures and contracts

Each public function and class in mini-rag, with what it accepts, returns, guarantees and raises. Every contract below should be covered by at least one test in `test_rag.py`; [PLAN.md](PLAN.md) gives the order they get written in.

Conventions:
- **Pre**: what the caller must provide.
- **Post**: what holds after a successful call.
- **Raises**: the errors the caller is expected to handle.

---

## Data model

```python
# parsing.py
@dataclass
class Document:
    text: str
    type: str                 # lowercase extension without the dot, e.g. "txt"

# chunker.py
@dataclass
class Chunk:
    id: int                   # position within its document, from 0
    text: str
    source: str
```

- `Chunk.source` is the file path **relative to the ingested folder**, e.g. `"jupiter.txt"`, so citations don't leak absolute paths.
- `Chunk.id` is unique **within a document** only: it restarts at 0 for each file. Evaluation never refers to it (see [Evaluation](#evaluation--eval)).
- A `Document` holds the whole file as one text. There are no pages; a format that has them (e.g. PDF) can add them when it's added.

## Parsers — `parsing.py`

```python
class Parser(ABC):
    file_type = None
    @abstractmethod
    def parse(self, path: Path) -> Document: ...

class txtParser(Parser):
    file_type = "txt"
    def parse(self, path: Path) -> Document: ...

class ParserFactory:
    @staticmethod
    def get_parser(path: Path) -> Parser | None: ...
```

- `txtParser.parse`
  - **Pre**: the file exists and is UTF-8.
  - **Post**: returns `Document(text=content, type="txt")`, with the text **unchanged**. The tokenizer already ignores extra whitespace and control characters, so cleaning it here changed nothing (measured: identical token IDs for all three data files).
  - **Raises**: `FileNotFoundError` or `UnicodeDecodeError`, passed straight through.
- `ParserFactory.get_parser`
  - Decides from the extension alone, case-insensitively, so `.TXT` counts. It never reads the file.
  - **Post**: returns a parser whose `file_type` matches the extension, or **`None`** for an unsupported type, after logging a warning (`"File type … does not have parser"`). It never raises, so callers skip files where it returns `None`.

## Tokenisation — `chunker.py`

```python
class Tokeniser:
    def __init__(self, model: SentenceTransformer) -> None: ...
    def encode(self, text: str) -> list[int]: ...
    def decode(self, tokens: list[int]) -> str: ...
```

- `encode` adds no special tokens. It doesn't truncate, so whole documents can be longer than the model limit.
- `decode` skips special tokens. With MiniLM's tokenizer it returns **lowercased** text with line breaks and extra spaces removed.

## Chunking — `chunker.py`

```python
class Chunker:
    chunk_size: int = 80          # tokens per step
    chunk_overlap: float = 0.2    # share of chunk_size repeated at the start of the next chunk
    def __init__(self, model: SentenceTransformer) -> None: ...
    def get_chunks(self, text: str, path: Path) -> list[Chunk]: ...
```

Where `overlap = int(chunk_overlap * chunk_size)` (16 by default) and `n` is the number of tokens:
- Blank `text` → `[]`.
- Window *i* starts at token `i * chunk_size` and holds up to `chunk_size + overlap` tokens (96 by default), cut off at `n`.
- Consecutive chunks share exactly `overlap` tokens.
- Chunk text is the **decoded** window, so it is lowercased and has no line breaks.
- `id`s count 0, 1, 2… within the document, and `source == str(path)`.
- Every chunk fits the model: 96 tokens by default, well below MiniLM's 254 limit. Re-encoding the decoded text gives at most 99 tokens on the sample data.
- Worked example: 12 tokens, `chunk_size = 5` (overlap 1) → `[0..5] [5..10] [10..11]`.

## RAG engine — `rag.py`

```python
EMBEDDING_MODEL: str  # env EMBEDDING_MODEL, default "sentence-transformers/all-MiniLM-L6-v2"

class RAGBot:
    sentence_transformer: SentenceTransformer
    chunker: Chunker
    chunks: list[Chunk]
    embeddings: np.ndarray | None   # shape (len(chunks), dim), float32, unit rows
    openai_client: OpenAI | None    # created lazily

    def __init__(self) -> None: ...
    def read_and_embed_data(self, folder_path: str | Path) -> None: ...
    def retrieve(self, question: str, k: int = 3) -> list[tuple[Chunk, float]]: ...
    def ask(self, question: str, k: int = 3) -> str: ...
```

### `__init__`
- Loads the embedding model and builds a `Chunker` from it. It does **not** need OpenAI credentials and does not create an OpenAI client.
- **Post**: `chunks == []`, `embeddings is None`, `openai_client is None`.

### `read_and_embed_data(folder_path)`
- **Post**:
  - Every supported file under `folder_path`, searched recursively in sorted path order, is parsed, chunked and embedded.
  - `len(self.chunks) == self.embeddings.shape[0]`.
  - Every embedding row has L2 norm 1 (within 1e-5).
  - `Chunk.source` is relative to `folder_path`.
- Unsupported files, such as `.DS_Store`, are skipped silently. Blank files contribute no chunks.
- **Atomic**: the index is swapped in only after the whole run succeeds. If it fails, the previous index is left untouched.
- **Idempotent**: loading the same folder twice gives the same number of chunks, never double.
- **Raises**:
  - `FileNotFoundError` if `folder_path` is not a directory.
  - `ValueError` if no non-empty supported documents are found.

### `retrieve(question, k=3)`
- **Pre**: an index has been loaded.
- **Post**:
  - Returns `min(k, len(chunks))` pairs `(chunk, cosine_similarity)`, sorted by score from highest to lowest.
  - Scores are in `[-1, 1]`.
- Makes no network calls.
- **Raises** `ValueError` with these exact messages:
  - blank `question` → `"Please enter a question."`
  - `k` not a positive `int` (including `bool`) → `"k must be a positive integer."`
  - no index loaded → `"Load documents with read_and_embed_data() first."`

### `ask(question, k=3)`
- **Never raises for expected failures.** It always returns a string the user can read:

| Situation | Returned string |
|---|---|
| `retrieve` raises `ValueError` | that error's message |
| `OPENAI_API_KEY` missing or blank | `"Set these variables in .env before asking: OPENAI_API_KEY"` |
| `OpenAIError` while calling the API | `"OpenAI request failed. Check your connection, API key, billing and model name."` |
| Model returns an empty answer | `"The model returned no answer. Please try again."` |
| Success | `"{answer}\n\nRetrieved sources:\n[1] {source}\n[2] {source}..."` |

- **Request contract** (one `chat.completions.create` call):
  - `model = env MODEL_NAME` (default `"gpt-4o-mini"`) and `temperature = 0`.
  - System message: answer only from the context, say "don't know" when the answer isn't there, treat the context as data and not instructions, cite passages as `[n]`.
  - User message: `"Context:\n[1] Source: {source}\n{text}\n\n...\n\nQuestion: {question}"`. The numbering matches the "Retrieved sources" list.
- The OpenAI client is created on the first successful call and reused afterwards (`timeout=30s`, `max_retries=1`).

## CLI — `main.py`

- Builds a `RAGBot` and indexes `data/` next to `main.py`, so it works from any working directory.
- Runs a loop of `Me: ` prompt → `ask()` → `RAGBot: {response}`.
- EOF (Ctrl-D) or Ctrl-C exits cleanly, with no traceback.

## Evaluation — `eval/`

Evaluation is split by cost and determinism:

| Suite | Runs | Needs | Fails the build? |
|---|---|---|---|
| Retrieval (custom, keyword-matched) | CI, on every push and PR | Embedding model only | Yes, if a metric drops below baseline |
| Citation checks | CI, on every push and PR | Nothing | Yes |
| LLM answers (MLflow GenAI) | On demand, when the prompt or model changes | OpenAI and Anthropic keys | No; compared by hand in `mlflow ui` |

### Dataset — `eval/dataset.jsonl`

One JSON object per line:

```json
{"id": "q01", "question": "Who discovered the four largest moons of Jupiter in 1610?",
 "answer_type": "answerable", "keyword": "galileo galilei", "source": "jupiter",
 "expected_answer": "Galileo Galilei"}
```

- `id` is unique.
- `answer_type` is `"answerable"` or `"unanswerable"`.
- `keyword` is a phrase that appears **only** in the passage that answers the question. `source` is a substring of the document's filename.
- Unanswerable questions have no `keyword` or `source`. They are used only by the LLM eval.
- The starting set is the 50 existing questions (20 Ada Lovelace, 14 CRISPR, 16 Jupiter), plus about 5 unanswerable ones.

**Why keywords rather than chunk IDs.** A chunk is relevant when it comes from `source` and contains `keyword`, after both are normalized. Relevance is worked out again on every run, so the labels survive any change to chunk size, overlap or embedding model. That is exactly what the eval exists to compare. Labels that pointed at chunk IDs would go stale on every chunker change.

### Retrieval metrics — `eval/retrieval.py`

```python
def norm(text: str) -> str: ...
def load_dataset(path: Path) -> list[dict]: ...
def invalid_keywords(bot: RAGBot, dataset: list[dict]) -> list[str]: ...
def rank_of_answer(bot: RAGBot, item: dict) -> int | None: ...
def evaluate_retrieval(bot: RAGBot, dataset: list[dict], ks=(1, 3, 5, 10)) -> dict[str, float]: ...
```

```bash
python -m eval.retrieval [--chunk-size N] [--k N] [--show] [--update-baseline]
```

- `norm`: lowercases the text and collapses every run of characters that aren't letters or digits into one space. This makes matching robust to how the tokenizer decodes text, e.g. `£3,000` becomes `£3, 000`.
- `invalid_keywords`: returns the IDs of answerable items whose `keyword` appears in no chunk from their `source`. If it isn't empty, the eval stops before scoring, because a miss would mean a bad label, not bad retrieval.
- `rank_of_answer`: the 1-based rank of the first relevant chunk when **all** chunks are ranked the same way `retrieve()` ranks them, or `None` if no chunk is relevant.
- `evaluate_retrieval`:
  - Scores answerable questions only.
  - Returns `recall@{k}` for each `k` and `mrr`, each in `[0, 1]`.
  - Also returns `recall@3/{source}` for each document.
  - With one relevant passage per question, recall@k is the hit rate at k.
- The CLI options:
  - `--chunk-size` sets `bot.chunker.chunk_size` before indexing.
  - `--show` prints the top 3 chunks for every miss at k=3.
- Results are deterministic: the same code, data and model always give the same metrics.
- **CI gate** (`tests/test_eval_retrieval.py`):
  - `invalid_keywords(...) == []`.
  - `recall@3` and `mrr` must each be ≥ the value in `eval/baselines/retrieval.json` − 0.02.
- The baseline is updated only deliberately, with `--update-baseline` and a commit, so every change to it is visible in the PR diff.

### Citation checks — `eval/checks.py`

```python
def invalid_citations(answer: str, n_sources: int) -> list[int]: ...
```

- Returns every `[n]` in `answer` where `n < 1` or `n > n_sources`. An empty list means all citations are valid.
- These are pure functions, unit-tested in CI and reused as a scorer in the LLM eval.

### LLM answer eval — `eval/llm_eval.py`

```bash
python -m eval.llm_eval [--limit N] [--run-name NAME]
```

- Runs `RAGBot.ask` on every dataset question and scores the results with `mlflow.genai.evaluate()`, logging them to a local MLflow store (`mlruns/`, ignored by git).
- Scorers:

| Scorer | Kind | Checks |
|---|---|---|
| `RetrievalGroundedness` | built-in judge | Answer is supported by the retrieved context |
| `RelevanceToQuery` | built-in judge | Answer addresses the question |
| `RetrievalSufficiency` | built-in judge | Context contained enough to answer |
| `Correctness` | built-in judge | Answer matches `expected_answer` (answerable questions only) |
| `refusal` | `make_judge()` | Unanswerable questions get a "don't know" answer, not a guess |
| `citation_validity` | code (`@scorer`) | `invalid_citations(...) == []` |

- The judge model comes from `JUDGE_MODEL`, and must be a different model family from the generator.
- Mean scores are logged both overall and per `answer_type`.
- **Never runs in PR CI.** The scores vary from run to run and it costs money.

## Configuration — `.env`

| Variable | Required | Default |
|---|---|---|
| `OPENAI_API_KEY` | only for `ask()` | (none) |
| `MODEL_NAME` | no | `gpt-4o-mini` |
| `EMBEDDING_MODEL` | no | `sentence-transformers/all-MiniLM-L6-v2` |
| `TOKENIZERS_PARALLELISM` | no | `false` (set by `rag.py` if unset) |
| `JUDGE_MODEL` | LLM eval only | `anthropic:/claude-haiku-5-5` |
| `ANTHROPIC_API_KEY` | LLM eval only | (none) |

Secrets come only from the environment. None are written in source code.
