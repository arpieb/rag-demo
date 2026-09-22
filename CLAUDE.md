# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A single-notebook RAG demo: open-access papers (by DOI) → Markdown → Chroma → a cited answer.
`rag_demo.py` is a marimo notebook — a plain, diffable Python file, not JSON. Each pipeline stage is
a function defined in its own cell; the cells that follow are just wiring, so stages can be reused or
tested independently of the notebook.

## Commands

```bash
uv sync                          # create/refresh .venv from uv.lock
uv run marimo edit rag_demo.py   # reactive editor
uv run marimo run rag_demo.py    # read-only app view
uv run python rag_demo.py        # execute the notebook headlessly, as a script
uv run marimo export script rag_demo.py   # cheap DAG check: fails on marimo dataflow errors
uv add <pkg>                     # add a dependency (updates pyproject.toml + uv.lock)
```

There is no test runner or linter configured; `uv run pytest` will not work until pytest is added.
uv pins CPython 3.14 from its own store, so always go through `uv run` rather than `python`.

## Runtime prerequisites

- **Ollama must be running locally.** mellea's `start_session()` defaults to `backend_name="ollama"`
  with `IBM_GRANITE_4_1_3B` (`granite4.1:3b`). Switch with `MELLEA_BACKEND` / `MELLEA_MODEL_ID`;
  the `openai`, `watsonx`, `bedrock`, `hf`, and `litellm` backends are also available, but `hf` and
  `litellm` need extras that are not installed.
- **`UNPAYWALL_EMAIL` must be set.** The Unpaywall API requires a contact address and explicitly
  rejects anything containing `example.com`, so the notebook's placeholder default will not work.

All configuration is env vars read in one cell near the top: `UNPAYWALL_EMAIL`, `RAG_DEMO_PDF_DIR`,
`RAG_DEMO_COLLECTION`, `RAG_DEMO_CHUNK_CHARS`, `RAG_DEMO_N_RESULTS`, `MELLEA_BACKEND`,
`MELLEA_MODEL_ID`.

## Library specifics worth knowing

- **Do not use `Unpywall.download_pdf_handle()`.** It builds its result as
  `BytesIO(bytearray(r.text, encoding='utf-8'))`, decoding binary PDF bytes as text and re-encoding
  them — the resulting file is corrupt and docling cannot parse it. Use `Unpywall.get_pdf_link(doi)`
  and fetch `response.content` directly, as `download_pdf()` does.
- **PDF extraction is `mellea.stdlib.components.docs.richdocument.RichDocument`** (the `mellea[docling]`
  extra), via `from_document_file(path, do_ocr=False)` → `.to_markdown()`. Keep `do_ocr=False` unless a
  PDF genuinely lacks a text layer: `True` downloads several hundred MB of OCR weights on first call.
- **Citations ride on mellea's `grounding_context`** — a `dict[str, str]` of named passages. The
  notebook keys them `source_1..source_N` and instructs the model to cite those keys, so the key
  naming and the instruction text are coupled. `instruct()` also runs `requirements` through its
  default `RejectionSamplingStrategy(loop_budget=2)`, which costs extra model calls per answer.
- **`FAILED. Valid: 1/2` in the logs is expected, not a regression.** granite4.1:3b is a conservative
  judge of the "only uses information from the sources" requirement and rejects answers that are in
  fact grounded. Sampling exhausts its loop budget and `select_from_failure` returns the best
  attempt. Don't "fix" it by deleting the requirement; a larger judge model is the real remedy.
- **Chroma uses `PersistentClient()` with no arguments**, which writes to `./chroma` (gitignored) and
  embeds with Chroma's bundled default model. Indexing uses `upsert`, not `add`, because marimo
  re-executes cells reactively and `add` would warn on duplicate IDs.

## Caching, and why re-runs are cheap

marimo runs every cell on startup, and docling conversion is slow, so two stages cache to disk:
downloaded PDFs are reused if present, and extracted Markdown is written beside each PDF as `.md`.
Delete `papers/` to force a refetch and re-extraction, `chroma/` to rebuild the index. Neither
directory is committed.

Generation is behind a `mo.ui.run_button` so editing the question does not fire the model on every
keystroke — keep it that way when adding cells downstream of the query input.
