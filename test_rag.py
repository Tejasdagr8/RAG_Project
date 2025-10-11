import os
import time
import requests
from main import api_crawl, api_index, api_ask

# Make sure OPENAI_API_KEY is set
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("Set your OPENAI_API_KEY environment variable first!")

# 1️⃣ Crawl a website
start_url = "https://en.wikipedia.org/wiki/FastAPI"  # example site
print(f"Crawling: {start_url} ...")
crawl_res = api_crawl(type('Req', (), {"start_url": start_url, "max_pages": 10, "crawl_delay_ms": 300, "max_depth": 1})())
print("Crawl finished:", crawl_res)

# 2️⃣ Build FAISS index
print("Building index...")
index_res = api_index(type('Req', (), {"chunk_size": 800, "chunk_overlap": 100, "embedding_model": "text-embedding-3-small"})())
print("Indexing finished:", index_res)

# 3️⃣ Ask a question
question = "What is FastAPI?"
print(f"Asking question: {question}")
ask_res = api_ask(type('Req', (), {"question": question, "top_k": 5, "generation_model": "gpt-4", "embedding_model": "text-embedding-3-small"})())
print("\nAnswer:\n", ask_res["answer"])
print("\nSources:")
for s in ask_res["sources"]:
    print("-", s["url"])
