import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from crawler import crawl
from indexer import build_index, query_index, load_index
import openai

# Ensure your OpenAI key is loaded
openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI(title="RAG Service")

# --- Request Schemas ---
class CrawlRequest(BaseModel):
    start_url: str
    max_pages: int = 30
    crawl_delay_ms: int = 500
    max_depth: int = 2

class IndexRequest(BaseModel):
    chunk_size: int = 800
    chunk_overlap: int = 100
    embedding_model: str = "text-embedding-3-small"

class AskRequest(BaseModel):
    question: str
    top_k: int = 5
    generation_model: str = "gpt-4"
    embedding_model: str = "text-embedding-3-small"

# --- Root ---
@app.get("/")
def root():
    return {"message": "RAG Service running"}

# --- Crawl Endpoint ---
@app.post("/crawl")
def api_crawl(req: CrawlRequest):
    try:
        return crawl(req.start_url, req.max_pages, req.crawl_delay_ms, req.max_depth)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Index Endpoint ---
@app.post("/index")
def api_index(req: IndexRequest):
    try:
        result = build_index(req.chunk_size, req.chunk_overlap, req.embedding_model)
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Ask Endpoint ---
@app.post("/ask")
def api_ask(req: AskRequest):
    if not openai.api_key:
        raise HTTPException(status_code=400, detail="OPENAI_API_KEY not set")

    try:
        hits = query_index(req.question, top_k=req.top_k, embedding_model=req.embedding_model)
        if not hits:
            return {"answer": "not found in crawled content", "sources": []}

        # Prepare context for LLM
        context_texts = [f"URL: {h['url']}\n{h['text']}" for h in hits]
        sources = [{"url": h["url"], "snippet": h["text"][:400]} for h in hits]
        context = "\n\n---\n\n".join(context_texts)
        prompt = f"""Use only the following context to answer the question. If answer not found, reply: not found in crawled content.

Context:
{context}

Question:
{req.question}

Answer concisely and list sources at end."""

        resp = openai.chat.completions.create(
            model=req.generation_model,
            messages=[
                {"role": "system", "content": "You are a strict grounded assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=512,
            temperature=0
        )
        answer = resp.choices[0].message.content.strip()
        return {"answer": answer, "sources": sources}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
