# mini-rag

A small retrieval-augmented generation (RAG) bot built from scratch: no LangChain, no vector database. It reads a folder of `.txt` files, chunks and embeds them locally, retrieves the most relevant passages for a question, and asks an OpenAI model for an answer grounded in, and citing, those passages.

```
Me: What are the four Galilean moons?
RAGBot: Io, Europa, Ganymede and Callisto [1].

Retrieved sources:
[1] jupiter.txt
[2] jupiter.txt
[3] jupiter.txt
```

## How it works

1. **Parse**: `parser.py` reads each `.txt` file as is. Other file types are skipped. `ParserFactory` is the place to add new formats.
2. **Chunk**: `chunker.py` splits text by *tokens*, not characters, using the embedding model's own tokenizer. Each chunk is 96 tokens: 80 new ones plus 16 repeated from the end of the previous chunk, so context at the boundaries isn't lost. Chunk text is decoded from the tokens, so it is lowercase. Each chunk keeps its source filename.
3. **Embed**: chunks are batch-encoded with [`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) and L2-normalized, so a dot product equals cosine similarity.
4. **Retrieve**: the question is embedded the same way, scored against every chunk with a single matrix product, and the top *k* (default 5) are used.
5. **Generate**: the retrieved chunks go into the prompt with numbered source labels. The system prompt tells the model to answer only from that context, treat it as data rather than instructions, cite passages as `[1]`, `[2]`, and say when the answer isn't there.

A NumPy matrix is enough at this scale. There's no persistence, reranker or conversation memory, by design.

## Project structure

```
mini-rag/
├── data/            # sample corpus: Ada Lovelace, Jupiter, CRISPR (from Wikipedia)
├── parser.py        # file → Document
├── chunker.py       # token-based overlapping chunker
├── rag.py           # RAGBot: index, retrieve, ask
├── models/          # bundled all-MiniLM-L6-v2 embedding model (~87 MB)
├── main.py          # interactive CLI
├── .vscode/         # run configuration and test settings for VS Code
├── eval/            # eval dataset, retrieval metrics, LLM judge eval
└── tests/           # pytest suite (real embedding model, mocked OpenAI)
```

## Setup

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then add your OpenAI or Azure OpenAI settings
```

`.env` variables. Fill in **one** provider:

| Variable | Provider | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | OpenAI | API key |
| `AZURE_OPENAI_ENDPOINT` | Azure | Resource endpoint; when it is set, Azure is used |
| `AZURE_OPENAI_API_KEY` | Azure | API key |
| `OPENAI_API_VERSION` | Azure | API version, e.g. `2024-10-21` |
| `MODEL_NAME` | both | Chat model (default `gpt-4.1-mini`); for Azure, the deployment name |

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

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The tests use the real embedding model and a mocked OpenAI client, so they make no paid API calls.

## Evaluation

| What | Tool | Where it runs |
|---|---|---|
| Retrieval quality: Recall@k and MRR over 50 questions, each with a keyword that marks its answer passage | plain Python (`eval/retrieval.py`) | CI, on every push. Fails if a metric drops below `eval/baselines/retrieval.json` |
| Citation validity | plain Python | CI, on every push |
| Answer groundedness, relevance, correctness, refusal on out-of-scope questions | [MLflow GenAI](https://mlflow.org/docs/latest/genai/) LLM judges | On demand: `python -m eval.llm_eval`, then `mlflow ui` |

CI needs no API keys. The LLM eval calls paid APIs and its scores vary from run to run, so it is run by hand whenever the prompt or model changes.

## Limitations

- Top-k retrieval always returns *something*. Similarity scores are not calibrated confidence.
- Grounding instructions reduce unsupported answers but don't guarantee correctness.
- The whole index lives in memory and is rebuilt on every start.

## Data

The sample texts in `data/` are excerpts from Wikipedia articles, licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
