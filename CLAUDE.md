# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

A RAG (Retrieval-Augmented Generation) app for asking questions about uploaded PDFs, with a Streamlit UI (`app.py`) as the primary interface and a terminal loop (`query.py`) as a CLI alternative. `storage.py` is the shared storage layer (SQLite for document bytes/chunk text, FAISS + BM25 for the search indices) that both `ingest.py` and `query.py` read/write through — there's no separate "build the index" step before you can chat; uploading a document ingests it immediately.

## Commands

```bash
pip install -r requirements.txt   # install deps
streamlit run app.py              # web UI: upload PDFs, manage documents, chat
python ingest.py                  # CLI alternative: bulk-(re)ingest every PDF in data/
python query.py                   # CLI alternative: terminal Q&A loop; type 'exit' to quit
```

There is no test suite, linter, or build step in this repo.

Documents can be added either by uploading through `app.py`, or by dropping a PDF in `data/` and running `python ingest.py` — both paths call the same `ingest.ingest_single_document(filename, file_bytes)` and end up in the same SQLite/FAISS/BM25 store, so they're interchangeable.

## Architecture

### Storage layer (`storage.py`)

Single source of truth for documents and chunks, and owner of the FAISS/BM25 index files:

- SQLite (`pdf_qna.db`, gitignored): `documents(id, filename, file_bytes BLOB, sha256 UNIQUE, uploaded_at, status)` and `chunks(id, document_id, page, text)`. `sha256` gives free dedup — `add_document` returns `None` for a file whose content already exists, and callers treat that as "already ingested."
- FAISS index is a `faiss.IndexIDMap(IndexFlatL2)` (not a plain flat index), stored at `vectorstore/faiss_index`. The chunk's own SQLite id doubles as its FAISS vector id, so `add_chunks_and_embed` can append new vectors with `add_with_ids` and `delete_document` can drop a document's vectors with `remove_ids` — neither needs a full rebuild.
- BM25 has no incremental API, so `rebuild_bm25()` rebuilds `BM25Okapi` from the full `chunks` table on every add/delete (cheap — tokenization + stats only, no embedding calls) and persists it to `vectorstore/faiss_index_bm25.pkl` alongside the ordered list of chunk ids it was built from (needed to map BM25's positional scores back to chunk ids).
- `get_chunks_by_ids(ids)` is how both FAISS and BM25 results get turned back into `{id, text, source, page}` dicts — `source` comes from a join to `documents.filename`.

### Ingest (`ingest.py`)

`ingest_single_document(filename, file_bytes)` is the one ingestion entry point, used by both `app.py` and the CLI: extract pages (`extract_pages_from_bytes`, PyMuPDF `fitz.open(stream=...)`) → chunk (`chunk_pages`, `RecursiveCharacterTextSplitter`, chunk_size=500/overlap=50) → embed (`create_embeddings`, `all-MiniLM-L6-v2`) → `storage.add_document` + `storage.add_chunks_and_embed`. The `__main__` block just walks `data/*.pdf` and calls this per file, for bulk/offline use.

### Query (`query.py`)

Hybrid retrieval with fusion and reranking, shared by `app.py` and the terminal loop:

1. `retrieve_context` — dense FAISS search (k=15); since the index is an `IndexIDMap`, the ids it returns are chunk ids directly.
2. `bm25_search` — keyword BM25 search (k=15), using the persisted `chunk_ids` list to map scores back to chunk ids.
3. `reciprocal_rank_fusion` — merges the two rankings (RRF, k=60), deduping by chunk id.
4. `rerank` — cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) narrows the fused list to `top_k=3`.
5. The top chunks' text is joined into `combined_context` and sent to an LLM, chosen by `USE_LOCAL_LLM` in `.env` (default true → `ask_ollama`, model `llama3.2`, needs Ollama running locally; set to `"false"` → `ask_hf`, Hugging Face Inference API, model `meta-llama/Llama-3.1-8B-Instruct`, needs `HF_API_TOKEN` in `.env`).
6. Answer plus deduplicated `(source, page)` citations are returned/printed.

The embed model and cross-encoder are lazily-initialized module-level singletons (`get_embed_model`, `get_reranker`), which also makes them effectively cached across Streamlit reruns within the same process.

### UI (`app.py`, Streamlit)

Sidebar has the uploader (dedup + ingest on upload) and a document list with per-document delete (calls `storage.delete_document`, which does the FAISS `remove_ids` + BM25 rebuild). Main area is a `st.chat_message`/`st.chat_input` loop that calls the same retrieval/rerank/LLM functions as `query.py`. Chat history is session-only display state (`st.session_state.history`) — each question is still answered independently; there's no conversation memory fed back into retrieval.

### Config

- `.env` (gitignored): `HF_API_TOKEN`, `USE_LOCAL_LLM`.
- Local LLM path requires Ollama installed and running with the `llama3.2` model pulled.
