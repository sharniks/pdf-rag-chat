import numpy as np
from sentence_transformers import SentenceTransformer
import ollama
from dotenv import load_dotenv
import os
import requests
from sentence_transformers import CrossEncoder

import storage

# -----------------------------------------------------------------------------
# Load environment variables (.env file)
# -----------------------------------------------------------------------------
# Used for securely storing secrets like Hugging Face API token.
load_dotenv()

USE_LOCAL_LLM = os.getenv("USE_LOCAL_LLM", "true").lower() == "false"


_embed_model = None
_reranker = None


def get_embed_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embed_model


def get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker


# -----------------------------------------------------------------------------
# Ask Hugging Face Hosted LLM
# -----------------------------------------------------------------------------
def ask_hf(
        query,
        context,
        max_new_tokens=256,
        temperature=0.0,
        model="meta-llama/Llama-3.1-8B-Instruct"):
    """
    Sends the retrieved context and user query to
    Hugging Face Inference API.

    Parameters
    ----------
    query : str
        User question.

    context : str
        Retrieved chunks from FAISS.

    max_new_tokens : int
        Maximum response length.

    temperature : float
        Controls randomness.
        Lower values = more deterministic answers.

    model : str
        Hugging Face hosted model.

    Returns
    -------
    str
        Generated answer.
    """

    # Read API token from environment variables.
    HF_TOKEN = os.getenv("HF_API_TOKEN")

    HF_API_URL = "https://router.huggingface.co/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {HF_TOKEN}"
    }

    if HF_TOKEN is None:
        raise RuntimeError(
            "HF_API_TOKEN not found. Please add it to your .env file."
        )

    # Prompt engineering:
    # We explicitly instruct the model to answer ONLY from
    # the retrieved context.
    prompt = f"""
    You are a helpful assistant.

    Answer the question using ONLY the provided context.

    If the answer is not present in the context,
    reply with "I don't know."

    Context:
    {context}

    Question:
    {query}

    Answer:
    """

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": max_new_tokens,
        "temperature": temperature
    }

    # Send request to Hugging Face API.
    response = requests.post(
        HF_API_URL,
        headers=headers,
        json=payload,
        timeout=120
    )
    response.raise_for_status()

    # Convert JSON response into Python dictionary.
    result = response.json()

    if (
        "choices" not in result
        or not result["choices"]
        or "message" not in result["choices"][0]
        or "content" not in result["choices"][0]["message"]
    ):
        raise RuntimeError(f"Unexpected response:\n{result}")

    # Extract generated answer.
    return result["choices"][0]["message"]["content"]


# -----------------------------------------------------------------------------
# Ask Local Ollama Model
# -----------------------------------------------------------------------------
def ask_ollama(query, context):
    """
    Sends the retrieved context to a locally running
    Ollama model.

    This avoids external API calls and keeps everything local.
    """

    prompt = f"""
    You are a helpful assistant.

    Answer the question using ONLY the provided context.

    If the answer is not in the context,
    say "I don't know."

    Context:
    {context}

    Question:
    {query}

    Answer:
    """

    response = ollama.chat(
        model="llama3.2",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]

# At query time


def retrieve_context(index, query, k=15):
    """Dense FAISS search. `index` is an IndexIDMap, so returned ids are chunk ids."""
    query_embedding = get_embed_model().encode([query])
    query_embedding = np.array(query_embedding).astype("float32")
    _distances, indices = index.search(query_embedding, k)
    ids = [int(i) for i in indices[0] if i != -1]
    return storage.get_chunks_by_ids(ids)


def bm25_search(bm25, chunk_ids, query, k=15):
    """Keyword search. `chunk_ids` maps BM25's positional scores back to chunk ids."""
    if bm25 is None or not chunk_ids:
        return []
    scores = bm25.get_scores(query.lower().split())
    top_idx = np.argsort(scores)[::-1][:k]
    top_ids = [chunk_ids[i] for i in top_idx]
    chunks_by_id = {c["id"]: c for c in storage.get_chunks_by_ids(top_ids)}
    return [
        (chunks_by_id[chunk_ids[i]], scores[i])
        for i in top_idx
        if chunk_ids[i] in chunks_by_id
    ]


def reciprocal_rank_fusion(dense_results, bm25_results, k=60):
    scores = {}
    lookup = {}

    for rank, chunk in enumerate(dense_results):
        cid = chunk["id"]
        scores[cid] = scores.get(cid, 0) + 1 / (k + rank + 1)
        lookup[cid] = chunk
    for rank, (chunk, _) in enumerate(bm25_results):
        cid = chunk["id"]
        scores[cid] = scores.get(cid, 0) + 1 / (k + rank + 1)
        lookup[cid] = chunk

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [lookup[cid] for cid, _ in ranked]


def rerank(query, candidates, top_k=3):
    pairs = [(query, c["text"]) for c in candidates]
    scores = get_reranker().predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [chunk for chunk, _ in ranked[:top_k]]


# -----------------------------------------------------------------------------
# Main Program
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    storage.init_db()
    index = storage.load_or_init_faiss_index()
    bm25, chunk_ids = storage.load_bm25()

    while True:

        # Read user question.
        q = input("\nAsk a question (or type 'exit'): ")

        # Exit chatbot.
        if q.lower() == "exit":
            break

        # ---------------------------------------------------------
        # Step 1 : Hybrid retrieval — dense + keyword
        # ---------------------------------------------------------
        dense_chunks = retrieve_context(index, q, k=15)
        bm25_chunks = bm25_search(bm25, chunk_ids, q, k=15)
        fused = reciprocal_rank_fusion(dense_chunks, bm25_chunks)

        if not fused:
            print("\nAnswer:\nI don't know. No documents have been ingested yet.")
            continue

        # ---------------------------------------------------------
        # Step 2 : Cross-encoder rerank down to final top-k
        # ---------------------------------------------------------
        top_chunks = rerank(q, fused[:15], top_k=3)
        combined_context = "\n\n".join(c["text"] for c in top_chunks)

        # ---------------------------------------------------------
        # Step 3 : Send context + question to LLM.
        # ---------------------------------------------------------
        if USE_LOCAL_LLM:
            answer = ask_ollama(q, combined_context)
        else:
            answer = ask_hf(q, combined_context)

        # ---------------------------------------------------------
        # Step 4 : Display response.
        # ---------------------------------------------------------
        print("\nAnswer:")
        print(answer)

        print("\nSources:")
        seen = set()
        for c in top_chunks:
            key = (c["source"], c["page"])
            if key not in seen:
                seen.add(key)
                print(f"  - {c['source']}, page {c['page']}")
