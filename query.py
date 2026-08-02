import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
import ollama
from dotenv import load_dotenv
import os
import requests
from sentence_transformers import CrossEncoder

# -----------------------------------------------------------------------------
# Load environment variables (.env file)
# -----------------------------------------------------------------------------
# Used for securely storing secrets like Hugging Face API token.
load_dotenv()

# Path where the FAISS index was created during the ingestion process.
INDEX_PATH = "./vectorstore/faiss_index"

USE_LOCAL_LLM = os.getenv("USE_LOCAL_LLM", "true").lower() == "false"


def load_index(index_path=INDEX_PATH):
    """
    Loads FAISS index, chunk metadata, and BM25 index together.
    Called explicitly from main() rather than at import time, so
    this module can be imported (e.g. by a test file or another
    script) without side effects — same idea as lazy-initializing
    a Spring bean instead of doing work in a static block.
    """
    index = faiss.read_index(index_path)
    with open(index_path + "_mapping.pkl", "rb") as f:
        chunks = pickle.load(f)
    with open(index_path + "_bm25.pkl", "rb") as f:
        bm25 = pickle.load(f)
    return index, chunks, bm25


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


def retrieve_context(index, chunks, query, k=15):
    query_embedding = get_embed_model().encode([query])
    query_embedding = np.array(query_embedding).astype("float32")
    distances, indices = index.search(query_embedding, k)
    return [chunks[i] for i in indices[0]]


def bm25_search(bm25, chunks, query, k=15):
    scores = bm25.get_scores(query.lower().split())
    top_idx = np.argsort(scores)[::-1][:k]
    return [(chunks[i], scores[i]) for i in top_idx]


def reciprocal_rank_fusion(dense_results, bm25_results, k=60):
    scores = {}
    lookup = {}

    def key(c):
        return (c["source"], c["page"], c["text"])

    for rank, chunk in enumerate(dense_results):
        ck = key(chunk)
        scores[ck] = scores.get(ck, 0) + 1 / (k + rank + 1)
        lookup[ck] = chunk
    for rank, (chunk, _) in enumerate(bm25_results):
        ck = key(chunk)
        scores[ck] = scores.get(ck, 0) + 1 / (k + rank + 1)
        lookup[ck] = chunk

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [lookup[ck] for ck, _ in ranked]


def rerank(query, candidates, top_k=3):
    pairs = [(query, c["text"]) for c in candidates]
    scores = get_reranker().predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [chunk for chunk, _ in ranked[:top_k]]


# -----------------------------------------------------------------------------
# Main Program
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    index, chunks, bm25 = load_index()

    while True:

        # Read user question.
        q = input("\nAsk a question (or type 'exit'): ")

        # Exit chatbot.
        if q.lower() == "exit":
            break

        # ---------------------------------------------------------
        # Step 1 : Hybrid retrieval — dense + keyword
        # ---------------------------------------------------------
        dense_chunks = retrieve_context(index, chunks, q, k=15)
        bm25_chunks = bm25_search(bm25, chunks, q, k=15)
        fused = reciprocal_rank_fusion(dense_chunks, bm25_chunks)
        top_chunks = rerank(q, fused[:15], top_k=3)

        # ---------------------------------------------------------
        # Step 2 : Cross-encoder rerank down to final top-k
        # ---------------------------------------------------------
        combined_context = "\n\n".join(c["text"] for c in top_chunks)

        # ---------------------------------------------------------
        # Step 2 : Send context + question to LLM.
        # ---------------------------------------------------------

        # Hugging Face Hosted LLM
        if USE_LOCAL_LLM:
            answer = ask_ollama(q, combined_context)
        else:
            answer = ask_hf(q, combined_context)

        # ---------------------------------------------------------
        # Step 3 : Display response.
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
        # ---------------------------------------------------------
        # Debugging (Optional)
        # Shows which chunks were retrieved.
        # Useful for understanding why the LLM answered a certain way.
        # ---------------------------------------------------------
        # print("\nRetrieved Chunks:\n")
        # for i, chunk in enumerate(top_chunks, start=1):
        #     print(f"{i}. {chunk[:200]}...")
