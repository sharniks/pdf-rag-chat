import os
import pickle
import hashlib
import logging
import sqlite3
from datetime import datetime, timezone

import numpy as np
import faiss
from rank_bm25 import BM25Okapi

logging.basicConfig(level=logging.INFO)

DB_PATH = "./pdf_qna.db"
INDEX_DIR = "./vectorstore"
FAISS_PATH = os.path.join(INDEX_DIR, "faiss_index")
BM25_PATH = os.path.join(INDEX_DIR, "faiss_index_bm25.pkl")

EMBEDDING_DIM = 384  # all-MiniLM-L6-v2


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    os.makedirs(INDEX_DIR, exist_ok=True)
    conn = get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            file_bytes BLOB NOT NULL,
            sha256 TEXT NOT NULL UNIQUE,
            uploaded_at TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'ingested'
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER NOT NULL REFERENCES documents(id),
            page INTEGER NOT NULL,
            text TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


# -----------------------------------------------------------------------------
# Documents
# -----------------------------------------------------------------------------
def add_document(filename, file_bytes):
    """
    Inserts a new document row. Returns None (and does nothing else) if a
    document with the same content already exists, so callers can skip
    re-embedding duplicates.
    """
    sha = hashlib.sha256(file_bytes).hexdigest()
    conn = get_connection()
    existing = conn.execute(
        "SELECT id FROM documents WHERE sha256 = ?", (sha,)
    ).fetchone()
    if existing:
        conn.close()
        return None

    cur = conn.execute(
        "INSERT INTO documents (filename, file_bytes, sha256, uploaded_at, status) "
        "VALUES (?, ?, ?, ?, ?)",
        (filename, file_bytes, sha, datetime.now(timezone.utc).isoformat(), "ingested"),
    )
    conn.commit()
    doc_id = cur.lastrowid
    conn.close()
    return doc_id


def list_documents():
    conn = get_connection()
    rows = conn.execute("""
        SELECT documents.id, documents.filename, documents.uploaded_at, documents.status,
               COUNT(chunks.id) AS chunk_count
        FROM documents
        LEFT JOIN chunks ON chunks.document_id = documents.id
        GROUP BY documents.id
        ORDER BY documents.uploaded_at DESC
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_document_bytes(doc_id):
    conn = get_connection()
    row = conn.execute(
        "SELECT filename, file_bytes FROM documents WHERE id = ?", (doc_id,)
    ).fetchone()
    conn.close()
    return (row["filename"], row["file_bytes"]) if row else None


def delete_document(doc_id):
    """
    Removes a document and its chunks from SQLite, drops its vectors from
    the FAISS index by id (no rebuild needed), and rebuilds BM25 from the
    remaining chunks.
    """
    conn = get_connection()
    chunk_rows = conn.execute(
        "SELECT id FROM chunks WHERE document_id = ?", (doc_id,)
    ).fetchall()
    chunk_ids = [r["id"] for r in chunk_rows]
    conn.execute("DELETE FROM chunks WHERE document_id = ?", (doc_id,))
    conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
    conn.commit()
    conn.close()

    if chunk_ids and os.path.exists(FAISS_PATH):
        index = faiss.read_index(FAISS_PATH)
        index.remove_ids(np.array(chunk_ids, dtype=np.int64))
        save_faiss_index(index)

    rebuild_bm25()
    logging.info(f"Deleted document {doc_id} ({len(chunk_ids)} chunks)")


# -----------------------------------------------------------------------------
# Chunks
# -----------------------------------------------------------------------------
def add_chunks_and_embed(doc_id, chunk_dicts, embeddings):
    """
    Persists chunk rows for a document, then adds their embeddings to the
    FAISS index using the chunk's own SQLite id as the vector id
    (IndexIDMap), and rebuilds BM25 over all chunks.
    """
    conn = get_connection()
    chunk_ids = []
    for c in chunk_dicts:
        cur = conn.execute(
            "INSERT INTO chunks (document_id, page, text) VALUES (?, ?, ?)",
            (doc_id, c["page"], c["text"]),
        )
        chunk_ids.append(cur.lastrowid)
    conn.commit()
    conn.close()

    embeddings = np.array(embeddings, dtype="float32")
    index = load_or_init_faiss_index(dim=embeddings.shape[1])
    index.add_with_ids(embeddings, np.array(chunk_ids, dtype=np.int64))
    save_faiss_index(index)

    rebuild_bm25()
    return chunk_ids


def get_chunks_by_ids(ids):
    """Returns chunk dicts (id, text, source, page) in the same order as ids."""
    ids = [int(i) for i in ids]
    if not ids:
        return []
    conn = get_connection()
    placeholders = ",".join("?" * len(ids))
    rows = conn.execute(
        f"""SELECT chunks.id, chunks.page, chunks.text, documents.filename AS source
            FROM chunks
            JOIN documents ON documents.id = chunks.document_id
            WHERE chunks.id IN ({placeholders})""",
        ids,
    ).fetchall()
    conn.close()
    by_id = {r["id"]: dict(r) for r in rows}
    return [by_id[i] for i in ids if i in by_id]


# -----------------------------------------------------------------------------
# FAISS index
# -----------------------------------------------------------------------------
def load_or_init_faiss_index(dim=EMBEDDING_DIM):
    if os.path.exists(FAISS_PATH):
        return faiss.read_index(FAISS_PATH)
    return faiss.IndexIDMap(faiss.IndexFlatL2(dim))


def save_faiss_index(index):
    os.makedirs(INDEX_DIR, exist_ok=True)
    faiss.write_index(index, FAISS_PATH)


# -----------------------------------------------------------------------------
# BM25 index
# -----------------------------------------------------------------------------
def rebuild_bm25():
    conn = get_connection()
    rows = conn.execute("SELECT id, text FROM chunks ORDER BY id").fetchall()
    conn.close()

    if not rows:
        bm25, chunk_ids = None, []
    else:
        tokenized = [r["text"].lower().split() for r in rows]
        bm25 = BM25Okapi(tokenized)
        chunk_ids = [r["id"] for r in rows]

    os.makedirs(INDEX_DIR, exist_ok=True)
    with open(BM25_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "chunk_ids": chunk_ids}, f)
    return bm25, chunk_ids


def load_bm25():
    if not os.path.exists(BM25_PATH):
        return None, []
    with open(BM25_PATH, "rb") as f:
        data = pickle.load(f)
    return data["bm25"], data["chunk_ids"]
