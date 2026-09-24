# rag-demo

A minimal retrieval-augmented generation (RAG) pipeline over open-access research papers, in a
single [marimo](https://marimo.io) notebook:

1. **Fetch** open-access PDFs for a list of DOIs via [Unpaywall](https://unpaywall.org)
2. **Extract** each PDF to Markdown with [mellea](https://github.com/generative-computing/mellea)'s
   docling-backed `RichDocument`
3. **Index** the text as chunks in a persistent [Chroma](https://www.trychroma.com) collection
4. **Retrieve** the chunks most relevant to a question
5. **Generate** an answer that cites its sources, using mellea's `instruct`

Each stage is a plain Python function in its own cell, so it can be reused or tested outside the
notebook. `rag_demo.py` is an ordinary Python file, not JSON, so it diffs cleanly.

## Prerequisites

- [uv](https://docs.astral.sh/uv/) — it installs the pinned Python (3.14) and all dependencies.
- [Ollama](https://ollama.com), running locally, with the default model pulled:

  ```bash
  ollama pull granite4.1:3b
  ```

  To use a different backend or model, see [Configuration](#configuration).
- A real email address for the Unpaywall API. It is required, and addresses at `example.com` are
  rejected.

## Quick start

```bash
uv sync
export UNPAYWALL_EMAIL=you@your.institution.edu
uv run marimo edit rag_demo.py
```

The notebook runs top to bottom on open: it downloads the papers, extracts and indexes them, and
retrieves chunks for the default question. Edit the **Question** box to re-run retrieval, then press
**Generate answer** to call the model. Generation sits behind that button so typing a question does
not call the model on every keystroke.

Other ways to run it:

```bash
uv run marimo run rag_demo.py   # read-only app view
uv run python rag_demo.py       # headless, as a plain script
```

## Configuration

All settings are environment variables:

| Variable               | Default          | Purpose                                                  |
| ---------------------- | ---------------- | -------------------------------------------------------- |
| `UNPAYWALL_EMAIL`      | *(placeholder)*  | Contact address for the Unpaywall API — **must be set**  |
| `RAG_DEMO_PDF_DIR`     | `./papers`       | Where PDFs and their extracted Markdown are stored       |
| `RAG_DEMO_COLLECTION`  | `papers`         | Chroma collection name                                   |
| `RAG_DEMO_CHUNK_CHARS` | `1200`           | Target chunk size, in characters                         |
| `RAG_DEMO_N_RESULTS`   | `5`              | Number of chunks retrieved per question                  |
| `MELLEA_BACKEND`       | `ollama`         | mellea backend: `ollama`, `openai`, `watsonx`, `bedrock` |
| `MELLEA_MODEL_ID`      | *(mellea default, `granite4.1:3b`)* | Model to use with that backend        |

The `hf` and `litellm` mellea backends also exist but need extras that are not installed.

To change the corpus, edit the `DOIS` list in the notebook. Any DOI with an open-access copy in
Unpaywall works; DOIs without one are skipped.

## Caching and re-runs

Downloading and PDF conversion are slow, so results are cached on disk:

- `papers/` holds the downloaded PDFs and, beside each, its extracted `.md`. Delete it to force a
  fresh download and extraction.
- `chroma/` holds the vector index. Delete it to rebuild the index. Indexing uses `upsert`, so
  re-running without deleting it is safe.

Both directories, and Unpaywall's `unpaywall_cache` file, are gitignored.

## Things to expect

- **`FAILED. Valid: 1/2` in the logs is normal.** mellea checks each answer against two
  requirements (it cites sources, and it uses only the sources). `granite4.1:3b` is a strict judge of
  the second one and often rejects answers that are properly grounded. mellea retries, then returns
  its best attempt, which is what the notebook shows. A larger model gives cleaner validation.
- **Each answer makes several model calls**, because of that retry-and-validate loop.
- **OCR is off.** Extraction uses the PDF's embedded text layer. Turning OCR on downloads several
  hundred MB of model weights the first time it runs, and these papers don't need it.

## Development

```bash
uv add <package>                         # add a dependency
uv run marimo export script rag_demo.py  # quick check for notebook dataflow errors
```

There is no test suite or linter configured yet.
