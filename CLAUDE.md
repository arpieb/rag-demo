# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Current state

This is a freshly scaffolded uv project: `pyproject.toml`, `uv.lock`, and `.venv/` only. There is no
application code, no README, no tests, and no git repository yet. Nothing below describes existing
modules — it describes the toolchain and the dependency set the project was scaffolded around.

## Toolchain

uv-managed, pinned to CPython 3.14 (`requires-python = ">=3.14"`). The interpreter lives under
`~/.local/share/uv/python/`, not the system Python — always go through `uv run` rather than invoking
`python` directly.

```bash
uv sync                      # create/refresh .venv from uv.lock
uv run python -c "..."       # run anything inside the env
uv add <pkg>                 # add a dependency (updates pyproject.toml + uv.lock)
uv run marimo edit app.py    # reactive notebook, edit mode
uv run marimo run app.py     # same file served as a read-only app
```

`pyproject.toml` declares no build backend, test runner, or lint config. Adding pytest/ruff means
adding the dependency and the corresponding `[tool.*]` section first; don't assume `uv run pytest`
works until it's been added.

## Dependencies and what they imply

- **mellea** (`>=0.7.0`) — generative-program library; the LLM layer. `mellea.start_session()`
  defaults to `backend_name="ollama"` with `model_id=IBM_GRANITE_4_1_3B`, so out of the box this
  expects a **local Ollama daemon**, not a hosted API. Other backends (`openai`, `hf`, `watsonx`,
  `litellm`, `bedrock`) are selectable via `start_session(backend_name=...)`; the `hf` and `litellm`
  ones need extras that are not currently installed. Session conveniences: `instruct()`, `chat()`,
  `query()`, `transform()`, plus `@generative` stubs.
- **chromadb** (`>=1.5.9`) — the vector store / retrieval half of the RAG demo.
- **unpywall** (`>=0.2.3`) — Unpaywall API client for locating open-access PDFs of papers. It
  requires an email address for API identification (Unpaywall's polite-pool requirement); expect it
  to be configured via `UNPAYWALL_EMAIL` or `UnpywallCredentials`. This is the corpus-acquisition
  side: fetch OA papers → chunk/embed into Chroma → answer with mellea.
- **marimo** (`>=0.24.2`) — the presentation layer. marimo notebooks are plain `.py` files with
  reactive cells, so they are diffable and importable; prefer putting demo flow in a marimo notebook
  and reusable logic in an adjacent module the notebook imports.

## Conventions worth preserving

Keep the pipeline stages (acquire → index → retrieve → generate) separable so the marimo notebook
can drive each independently — that reactivity is the reason marimo is here rather than Jupyter.
