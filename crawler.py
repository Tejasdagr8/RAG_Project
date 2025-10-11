import requests
from bs4 import BeautifulSoup
import tldextract
import time
from utils import save_json

def crawl(start_url, max_pages=30, crawl_delay_ms=500, max_depth=2):
    visited = set()
    pages = []

    def crawl_recursive(url, depth):
        if url in visited or len(pages) >= max_pages or depth > max_depth:
            return
        visited.add(url)
        try:
            res = requests.get(url, timeout=5)
            if res.status_code != 200:
                return
            soup = BeautifulSoup(res.text, "html.parser")
            text = soup.get_text(separator="\n")
            pages.append({"url": url, "text": text})
            # find links
            for a in soup.find_all("a", href=True):
                link = a["href"]
                if link.startswith("/"):
                    base = tldextract.extract(url)
                    link = f"https://{base.domain}.{base.suffix}{link}"
                elif not link.startswith("http"):
                    continue
                crawl_recursive(link, depth + 1)
            time.sleep(crawl_delay_ms / 1000.0)
        except:
            return

    crawl_recursive(start_url, 0)
    save_json("crawled_pages.json", pages)
    return {"pages_crawled": len(pages)}
