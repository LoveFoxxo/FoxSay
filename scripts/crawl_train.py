"""
crawl_train.py -- grow weights by continuously crawling web pages and feeding
their knowledge into the model, WITHOUT building a local dataset first.

Method (streaming crawl -> clean -> train):
  1. BFS crawl from user-supplied seed URLs: fetch, honor robots.txt, polite
     delay, max-pages / max-depth bounds, timeouts and retries.
  2. Clean: strip scripts/styles/tags, drop boilerplate (nav/footer), keep
     paragraphs above a length floor, dedupe by content hash (the crawl IS
     the dataset -- nothing big is written to disk, only a small dedupe set).
  3. Train immediately: tokenized text is batched in memory and consumed by
     the model as next-token prediction as soon as a batch is full.
  4. Every `save_every` steps, weights are written as 2/3/4 region shards
     via FoxSay.brain_shards.

Usage:
  python scripts/crawl_train.py --seeds "https://<site>/start" \
      --max_pages 200 --n_shards 3 --save_every 25
"""

from __future__ import annotations

import argparse
import html as _html
import os
import re
import time
import urllib.parse
import urllib.request
import urllib.robotparser

import torch
import torch.nn.functional as F

# make the repo root importable when run as `python scripts/crawl_train.py`
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

_TAG_RE = re.compile(r"<script[\s\S]*?</script>|<style[\s\S]*?</style>|<[^>]+>")
_WS_RE = re.compile(r"\s+")
_BOILER_RE = re.compile(r"(cookie|privacy|subscribe|advertisement)", re.I)

UA = "Mozilla/5.0 (compatible; FoxSay-crawler/0.1; research)"
MAX_PAGE_KB = 512


# --- crawling ---------------------------------------------------------------

def _robots_allowed(url: str, ua: str = "FoxSay-crawler") -> bool:
    try:
        rp = urllib.robotparser.RobotFileParser()
        rp.set_url(urllib.parse.urljoin(url, "/robots.txt"))
        rp.read()
        return rp.can_fetch(ua, url)
    except Exception:
        return True  # no robots.txt -> allow


def fetch(url: str, timeout: int = 15) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        if r.headers.get_content_type() not in ("text/html", "text/plain"):
            return ""
        raw = r.read(MAX_PAGE_KB * 1024)
        return raw.decode("utf-8", errors="ignore")


def clean_html(html_text: str):
    text = _TAG_RE.sub("\n", html_text)
    text = _html.unescape(text)
    paras = []
    for p in text.split("\n"):
        p = _WS_RE.sub(" ", p).strip()
        if len(p) >= 40 and not _BOILER_RE.search(p):
            paras.append(p)
    return "\n".join(paras)


def extract_links(html_text: str, base_url: str):
    links = set()
    for m in re.finditer(r'href=["\']([^"\'#]+)', html_text):
        u = urllib.parse.urljoin(base_url, m.group(1))
        if u.startswith("http") and not u.endswith((".png", ".jpg", ".pdf", ".zip")):
            links.add(u)
    return links


def crawl(seeds, max_pages: int, max_depth: int = 2, delay: float = 0.5):
    """BFS generator yielding cleaned page texts; nothing big hits disk."""
    seen = set()
    frontier = [(s, 0) for s in seeds]
    fetched = 0
    while frontier and fetched < max_pages:
        url, depth = frontier.pop(0)
        if url in seen or depth > max_depth:
            continue
        seen.add(url)
        try:
            if not _robots_allowed(url):
                continue
            raw = fetch(url)
        except Exception:
            continue
        if not raw:
            continue
        fetched += 1
        yield clean_html(raw)
        if depth < max_depth:
            for link in extract_links(raw, url):
                if link not in seen:
                    frontier.append((link, depth + 1))
        time.sleep(delay)  # polite crawl


# --- streaming trainer --------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="*", default=["https://example.com"],
                    help="start URLs for the crawl (required in practice)")
    ap.add_argument("--max_pages", type=int, default=200)
    ap.add_argument("--max_depth", type=int, default=2)
    ap.add_argument("--delay", type=float, default=0.5)
    ap.add_argument("--save_every", type=int, default=25)
    ap.add_argument("--n_shards", type=int, default=3, choices=(2, 3, 4))
    ap.add_argument("--out_dir", default="runs/crawl_train")
    ap.add_argument("--batch_tokens", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n_layer", type=int, default=4)
    ap.add_argument("--n_embd", type=int, default=256)
    ap.add_argument("--vocab_size", type=int, default=50257,
                    help="must match the tokenizer (gpt2 = 50257; change both together)")
    cfg = ap.parse_args()

    import tiktoken
    tok = tiktoken.get_encoding("gpt2")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from FoxSay.gpt import GPT, GPTConfig
    m = GPT(GPTConfig(n_layer=cfg.n_layer, n_embd=cfg.n_embd,
                      n_head=4, n_kv_head=4, vocab_size=cfg.vocab_size,
                      sequence_len=512))
    m.to(device); m.init_weights(); m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=cfg.lr)

    from FoxSay import brain_shards
    os.makedirs(cfg.out_dir, exist_ok=True)

    buf = []          # token buffer, flushed to a batch when full
    n_buf_tokens = 0
    step = 0
    seen_text = set()

    try:
        for text in crawl(cfg.seeds, cfg.max_pages, cfg.max_depth, cfg.delay):
            if not text:
                continue
            h = hash(text)
            if h in seen_text:
                continue
            seen_text.add(h)
            ids = tok.encode(text)[:512]
            ids = [i for i in ids if i < cfg.vocab_size]  # clamp to model vocab
            if len(ids) < 8:
                continue
            buf.extend(ids)
            n_buf_tokens += len(ids)
            # flush buffer as soon as a batch is ready: train while crawling
            while n_buf_tokens >= cfg.batch_tokens:
                flat = torch.tensor(buf[:cfg.batch_tokens + 1], device=device)
                x = flat[:-1].unsqueeze(0); y = flat[1:].unsqueeze(0)
                logits = m(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.reshape(-1))
                loss.backward()
                opt.step(); opt.zero_grad()
                step += 1
                print(f"step {step} loss {float(loss):.4f} "
                      f"(crawled {len(seen_text)} pages)")
                if step % cfg.save_every == 0:
                    brain_shards.save_shards(m, cfg.out_dir, cfg.n_shards, step,
                                             optimizer=opt)
                    print(f"[shards] saved {cfg.n_shards} regions @ step {step}")
                buf = buf[cfg.batch_tokens:]
                n_buf_tokens -= cfg.batch_tokens
    except KeyboardInterrupt:
        print("interrupted; saving shards before exit")
    finally:
        brain_shards.save_shards(m, cfg.out_dir, cfg.n_shards, step, optimizer=opt)
        print(f"done: {step} steps, {cfg.n_shards} region shards in {cfg.out_dir}")


if __name__ == "__main__":
    main()
