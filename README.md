# RAG Project

A clean **Retrieval-Augmented Generation** teaching service: crawl → chunk → embed → ask, with source-grounded answers.

Built with **FastAPI**, **OpenAI embeddings**, and **FAISS**.

## What it demonstrates

- Configurable website crawl (depth / page limits)
- Text chunking with overlap
- Vector index via FAISS (`text-embedding-3-small`)
- Q&A with cited sources (`gpt-4` / `gpt-3.5-turbo`)

## Stack

| Layer | Choice |
| --- | --- |
| API | FastAPI + Uvicorn |
| Embeddings | OpenAI `text-embedding-3-small` |
| Index | FAISS (CPU) |
| Generation | OpenAI Chat Completions |

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # if present; otherwise export OPENAI_API_KEY
export OPENAI_API_KEY="YOUR_KEY_HERE"
uvicorn main:app --reload
```

Requires **Python 3.12+**.

## API

### Health

```bash
curl http://127.0.0.1:8000/
```

### Crawl

```bash
curl -X POST "http://127.0.0.1:8000/crawl" \
  -H "Content-Type: application/json" \
  -d '{"start_url":"https://example.com","max_pages":5,"crawl_delay_ms":500,"max_depth":2}'
```

### Index

```bash
curl -X POST "http://127.0.0.1:8000/index" \
  -H "Content-Type: application/json" \
  -d '{"chunk_size":800,"chunk_overlap":100,"embedding_model":"text-embedding-3-small"}'
```

### Ask

```bash
curl -X POST "http://127.0.0.1:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question":"What is example.com?","top_k":3,"generation_model":"gpt-4","embedding_model":"text-embedding-3-small"}'
```

## Tests

```bash
pytest test_rag.py -q
```

## Related

For a production-minded medical RAG with trusted sources and citation anchoring, see **[MedTruth_AI](https://github.com/Tejasdagr8/MedTruth_AI)**.

## Author

[Tejas Melkote](https://github.com/Tejasdagr8) · [Portfolio](https://tejas-melkote.vercel.app/)
