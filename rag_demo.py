import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # RAG over open-access papers

    A minimal retrieval-augmented generation pipeline, one stage per section:

    1. **Fetch** open-access PDFs for a list of DOIs (unpywall)
    2. **Extract** each PDF to Markdown (mellea's docling-backed `RichDocument`)
    3. **Index** the documents as chunks in a persistent Chroma collection
    4. **Retrieve** the chunks relevant to a query
    5. **Generate** an answer with citations (mellea `instruct`)

    Every stage is a plain function, so the cells below are just wiring.
    """
    )
    return


@app.cell
def _():
    import os
    from pathlib import Path

    import chromadb
    import requests
    from mellea import start_session
    from mellea.stdlib.chunking import ParagraphChunker
    from mellea.stdlib.components.docs.richdocument import RichDocument
    from unpywall import Unpywall
    from unpywall.utils import UnpywallCredentials

    return (
        ParagraphChunker,
        Path,
        RichDocument,
        Unpywall,
        UnpywallCredentials,
        chromadb,
        os,
        requests,
        start_session,
    )


@app.cell
def _(Path, os):
    # Configuration: every knob is an env var with a demo default.
    # UNPAYWALL_EMAIL is the only one you must really set -- the Unpaywall API
    # requires a contact address, and it rejects anything at example.com.
    UNPAYWALL_EMAIL = os.environ.get("UNPAYWALL_EMAIL", "your.name@your.institution.edu")
    PDF_DIR = Path(os.environ.get("RAG_DEMO_PDF_DIR", "./papers"))
    COLLECTION_NAME = os.environ.get("RAG_DEMO_COLLECTION", "papers")
    CHUNK_CHARS = int(os.environ.get("RAG_DEMO_CHUNK_CHARS", "1200"))
    N_RESULTS = int(os.environ.get("RAG_DEMO_N_RESULTS", "5"))

    # mellea defaults to a local Ollama daemon serving granite4.1:3b.
    # Leave MELLEA_MODEL_ID empty to use whatever mellea's default is.
    MELLEA_BACKEND = os.environ.get("MELLEA_BACKEND", "ollama")
    MELLEA_MODEL_ID = os.environ.get("MELLEA_MODEL_ID", "")

    return (
        CHUNK_CHARS,
        COLLECTION_NAME,
        MELLEA_BACKEND,
        MELLEA_MODEL_ID,
        N_RESULTS,
        PDF_DIR,
        UNPAYWALL_EMAIL,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 1. Fetch open-access PDFs by DOI""")
    return


@app.cell
def _():
    # The corpus. Any DOI with an open-access copy in Unpaywall works here.
    DOIS = [
        "10.1371/journal.pcbi.1007084",   # ML/DL meets genome-scale metabolic modeling
        "10.1186/s12859-021-04344-9",     # CellProfiler 4
        "10.1186/s12859-020-3446-5",      # WGCNA-M pathway identification
    ]
    return (DOIS,)


@app.cell
def _(Path, Unpywall, UnpywallCredentials, requests):
    def configure_unpaywall(email: str) -> None:
        """Register the contact address the Unpaywall API requires."""
        UnpywallCredentials(email)  # also exports UNPAYWALL_EMAIL to the environment

    def paper_metadata(doi: str) -> dict:
        """Bibliographic fields for a DOI, carried through to the citations."""
        record = Unpywall.get_json(doi, errors="ignore") or {}
        return {
            "doi": doi,
            "title": record.get("title") or doi,
            "year": str(record.get("year") or ""),
            "journal": record.get("journal_name") or "",
        }

    def download_pdf(doi: str, dest_dir: Path) -> Path | None:
        """Save the best open-access PDF for a DOI; None when there isn't one.

        Fetches the bytes directly rather than via Unpywall.download_pdf_handle(),
        which round-trips the response through str() and corrupts binary PDFs.
        """
        url = Unpywall.get_pdf_link(doi)
        if not url:
            return None
        dest_dir.mkdir(parents=True, exist_ok=True)
        path = dest_dir / f"{doi.replace('/', '_')}.pdf"
        if path.exists():  # cached from a previous run
            return path
        response = requests.get(url, timeout=60, headers={"User-Agent": "rag-demo"})
        response.raise_for_status()
        path.write_bytes(response.content)
        return path

    def fetch_papers(dois: list[str], dest_dir: Path) -> list[dict]:
        """Download every DOI that has an OA PDF, keeping its metadata alongside."""
        papers = []
        for doi in dois:
            path = download_pdf(doi, dest_dir)
            if path is None:
                continue
            papers.append({"path": path, **paper_metadata(doi)})
        return papers

    return configure_unpaywall, fetch_papers


@app.cell
def _(DOIS, PDF_DIR, UNPAYWALL_EMAIL, configure_unpaywall, fetch_papers, mo):
    configure_unpaywall(UNPAYWALL_EMAIL)
    papers = fetch_papers(DOIS, PDF_DIR)

    mo.md(
        f"Fetched **{len(papers)}** of {len(DOIS)} DOIs into `{PDF_DIR}`:\n\n"
        + "\n".join(f"- {p['title']} ({p['year']})" for p in papers)
    )
    return (papers,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 2. Extract each PDF to a document

    `RichDocument` is mellea's docling-backed extractor: it converts the PDF and
    renders it as Markdown. `do_ocr=False` keeps it to the embedded text layer, which
    is what these papers have and avoids pulling down OCR weights.
    """
    )
    return


@app.cell
def _(Path, RichDocument):
    def extract_document(path: Path) -> str:
        """Convert one PDF to Markdown, caching the result next to the PDF."""
        cached = path.with_suffix(".md")
        if cached.exists():
            return cached.read_text()
        text = RichDocument.from_document_file(str(path), do_ocr=False).to_markdown()
        cached.write_text(text)
        return text

    def extract_documents(papers: list[dict]) -> list[dict]:
        """Attach the extracted text to each paper's metadata."""
        return [{**paper, "text": extract_document(paper["path"])} for paper in papers]

    return (extract_documents,)


@app.cell
def _(extract_documents, mo, papers):
    documents = extract_documents(papers)

    mo.md(
        "Extracted:\n\n"
        + "\n".join(f"- `{d['doi']}` — {len(d['text']):,} characters" for d in documents)
    )
    return (documents,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 3. Index the documents in Chroma

    A default `PersistentClient()` writes to `./chroma` and embeds with Chroma's
    built-in model, so there is nothing to configure.
    """
    )
    return


@app.cell
def _(chromadb):
    client = chromadb.PersistentClient()
    return (client,)


@app.cell
def _(ParagraphChunker):
    def chunk_text(text: str, max_chars: int) -> list[str]:
        """Split on paragraph boundaries, then pack paragraphs into ~max_chars windows."""
        chunker = ParagraphChunker()
        paragraphs = chunker.split(text) + chunker.flush(text)

        chunks: list[str] = []
        current = ""
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            # A paragraph longer than the budget is broken at the last space
            # before it, so chunks never end mid-word.
            while len(paragraph) > max_chars:
                cut = paragraph.rfind(" ", 0, max_chars)
                if cut <= 0:  # nothing to break on, e.g. one very long token
                    cut = max_chars
                if current:
                    chunks.append(current)
                    current = ""
                chunks.append(paragraph[:cut].strip())
                paragraph = paragraph[cut:].lstrip()
            if current and len(current) + len(paragraph) + 2 > max_chars:
                chunks.append(current)
                current = paragraph
            else:
                current = f"{current}\n\n{paragraph}" if current else paragraph
        if current:
            chunks.append(current)
        return chunks

    return (chunk_text,)


@app.cell
def _(chunk_text):
    def index_documents(client, collection_name: str, documents: list[dict], max_chars: int):
        """Chunk every document into the collection. Upserts, so re-running is safe."""
        collection = client.get_or_create_collection(collection_name)
        for document in documents:
            chunks = chunk_text(document["text"], max_chars)
            collection.upsert(
                ids=[f"{document['doi']}::{i}" for i in range(len(chunks))],
                documents=chunks,
                metadatas=[
                    {
                        "doi": document["doi"],
                        "title": document["title"],
                        "year": document["year"],
                        "chunk": i,
                    }
                    for i in range(len(chunks))
                ],
            )
        return collection

    return (index_documents,)


@app.cell
def _(CHUNK_CHARS, COLLECTION_NAME, client, documents, index_documents, mo):
    collection = index_documents(client, COLLECTION_NAME, documents, CHUNK_CHARS)

    mo.md(f"Collection `{collection.name}` holds **{collection.count()}** chunks.")
    return (collection,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 4. Retrieve the chunks relevant to a query""")
    return


@app.cell
def _():
    def retrieve(collection, query: str, n_results: int) -> list[dict]:
        """Nearest chunks for a query, flattened into one record per hit."""
        result = collection.query(query_texts=[query], n_results=n_results)
        return [
            {"text": text, "metadata": metadata, "distance": distance}
            for text, metadata, distance in zip(
                result["documents"][0],
                result["metadatas"][0],
                result["distances"][0],
            )
        ]

    return (retrieve,)


@app.cell
def _(mo):
    query_input = mo.ui.text(
        value="What software tools are described, and what problems do they solve?",
        label="Question",
        full_width=True,
    )
    query_input
    return (query_input,)


@app.cell
def _(N_RESULTS, collection, mo, query_input, retrieve):
    hits = retrieve(collection, query_input.value, N_RESULTS)

    mo.md(
        "Retrieved:\n\n"
        + "\n".join(
            f"- **source_{i}** — {h['metadata']['title']} "
            f"(distance {h['distance']:.3f})"
            for i, h in enumerate(hits, start=1)
        )
    )
    return (hits,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 5. Generate an answer with citations

    Each retrieved chunk goes into mellea's `grounding_context` under a `source_N`
    key, and the instruction asks the model to cite those keys. `requirements` are
    checked by mellea's default rejection sampling.

    Expect to see `FAILED. Valid: 1/2` in the logs: a 3B model is a conservative
    judge of its own grounding, so the second requirement often fails validation
    even when the answer is properly sourced. mellea retries within its loop budget
    and then returns its best attempt, which is what renders below.
    """
    )
    return


@app.cell
def _(MELLEA_BACKEND, MELLEA_MODEL_ID, start_session):
    session_kwargs = {"backend_name": MELLEA_BACKEND}
    if MELLEA_MODEL_ID:
        session_kwargs["model_id"] = MELLEA_MODEL_ID

    m = start_session(**session_kwargs)
    return (m,)


@app.cell
def _():
    def citation_label(metadata: dict) -> str:
        """One-line bibliographic label shown to the model and in the reference list."""
        year = f", {metadata['year']}" if metadata.get("year") else ""
        return f"{metadata['title']}{year} (doi:{metadata['doi']})"

    def build_grounding_context(hits: list[dict]) -> dict[str, str]:
        """Name each chunk source_1..source_N so the model has a key to cite."""
        return {
            f"source_{i}": f"{citation_label(hit['metadata'])}\n\n{hit['text']}"
            for i, hit in enumerate(hits, start=1)
        }

    def answer_with_citations(session, query: str, hits: list[dict]) -> str:
        """Ask the model to answer strictly from the retrieved chunks, with citations."""
        result = session.instruct(
            "Answer the question using only the provided sources. "
            "Cite every claim with the key of the source it came from, like [source_1]. "
            "If the sources do not answer the question, say so.\n\n"
            f"Question: {query}",
            grounding_context=build_grounding_context(hits),
            requirements=[
                "The answer cites its sources using bracketed keys like [source_1].",
                "The answer only uses information that appears in the provided sources.",
            ],
        )
        return result.value or ""

    return answer_with_citations, citation_label


@app.cell
def _(mo):
    run_button = mo.ui.run_button(label="Generate answer")
    run_button
    return (run_button,)


@app.cell
def _(answer_with_citations, citation_label, hits, m, mo, query_input, run_button):
    mo.stop(not run_button.value, mo.md("*Press **Generate answer** to run the model.*"))

    answer = answer_with_citations(m, query_input.value, hits)
    references = "\n".join(
        f"- **[source_{i}]** {citation_label(h['metadata'])}"
        for i, h in enumerate(hits, start=1)
    )

    mo.md(f"### Answer\n\n{answer}\n\n### References\n\n{references}")
    return


if __name__ == "__main__":
    app.run()
