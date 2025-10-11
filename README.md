# RAG Service

A simple **Retrieval-Augmented Generation (RAG)** service using FastAPI, OpenAI embeddings, and FAISS.  
Crawl websites, index content, and ask questions with source grounding.

---

## Features

- Web crawling with configurable depth and pages
- Text chunking & OpenAI embeddings (`text-embedding-3-small`)
- FAISS index for fast vector search
- Q&A using OpenAI ChatCompletion (`gpt-4`/`gpt-3.5-turbo`)
- Returns sources with answers

---

## Requirements

- Python 3.12+
- Packages: `fastapi`, `uvicorn`, `openai`, `faiss-cpu`, `requests`
- OpenAI API key

Install dependencies:

```bash
pip install -r requirements.txt

##Setup
python -m venv .venv
source .venv/bin/activate
export OPENAI_API_KEY="YOUR_KEY_HERE"
uvicorn main:app --reload


##API endpoints
curl http://127.0.0.1:8000/

##POST /crawl

Crawl a website and save pages.
curl -X POST "http://127.0.0.1:8000/crawl" \
-H "Content-Type: application/json" \
-d '{"start_url":"https://example.com","max_pages":5,"crawl_delay_ms":500,"max_depth":2}'


##POST /index

Build FAISS embeddings index.
curl -X POST "http://127.0.0.1:8000/index" \
-H "Content-Type: application/json" \
-d '{"chunk_size":800,"chunk_overlap":100,"embedding_model":"text-embedding-3-small"}'

##POST /ask

Ask questions based on indexed content.
curl -X POST "http://127.0.0.1:8000/ask" \
-H "Content-Type: application/json" \
-d '{"question":"What is example.com?","top_k":3,"generation_model":"gpt-4","embedding_model":"text-embedding-3-small"}'

##Notes

Ensure OPENAI_API_KEY is set

Crawled pages → crawled_pages.json

Indexing may take time depending on page count

top_k controls the number of chunks used for Q&A