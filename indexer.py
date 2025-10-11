import os
import json
import faiss
import numpy as np
import openai

# Make sure your API key is set in environment
openai.api_key = os.getenv("OPENAI_API_KEY")

# File paths
INDEX_FILE = "faiss.index"
CHUNKS_FILE = "chunks.json"
CRAWLED_FILE = "crawled_pages.json"

# --- Utility functions for JSON ---
def load_json(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

# --- Embeddings using new OpenAI API ---
def get_embedding(text, model="text-embedding-3-small"):
    if not text:
        return []
    resp = openai.embeddings.create(
        model=model,
        input=text
    )
    return resp.data[0].embedding

def embed_texts(texts, model="text-embedding-3-small"):
    embeddings = []
    for t in texts:
        embeddings.append(get_embedding(t, model=model))
    return np.array(embeddings, dtype="float32")

# --- Build FAISS index ---
def build_index(chunk_size=800, chunk_overlap=100, embedding_model="text-embedding-3-small"):
    pages = load_json(CRAWLED_FILE)
    if not pages:
        return {"error": f"{CRAWLED_FILE} not found or empty. Run /crawl first."}

    chunks = []
    texts = []

    for p in pages:
        text = p.get("text", "")
        url = p.get("url", "")
        for i in range(0, len(text), chunk_size - chunk_overlap):
            chunk_text = text[i:i+chunk_size]
            chunks.append({"text": chunk_text, "url": url})
            texts.append(chunk_text)

    # Embed all chunks
    embeddings = embed_texts(texts, model=embedding_model)
    if embeddings.size == 0:
        return {"error": "No embeddings generated"}

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, INDEX_FILE)
    save_json(CHUNKS_FILE, chunks)
    return {"chunks_indexed": len(chunks)}

# --- Load FAISS index ---
def load_index():
    if not os.path.exists(INDEX_FILE) or not os.path.exists(CHUNKS_FILE):
        return None, []
    index = faiss.read_index(INDEX_FILE)
    chunks = load_json(CHUNKS_FILE)
    return index, chunks

# --- Query the index ---
def query_index(question, top_k=5, embedding_model="text-embedding-3-small"):
    index, chunks = load_index()
    if index is None or not chunks:
        return []

    q_emb = np.array([get_embedding(question, model=embedding_model)], dtype="float32")
    D, I = index.search(q_emb, top_k)
    results = [chunks[i] for i in I[0]]
    return results
