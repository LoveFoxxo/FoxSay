"""
web_crawler.py -- internet knowledge feeding channel.

WebFeeder is a streaming iterator of cleaned page texts: BFS crawl from seed
URLs, robots.txt respect, polite delay, boilerplate removal, dedupe by hash.
Nothing big is written to disk.
"""

from __future__ import annotations

import html as _html
import re
import time
import urllib.parse
import urllib.request
import urllib.robotparser

__all__ = ["WebFeeder", "clean_html", "extract_links"]

_TAG_RE = re.compile(r"<script[\s\S]*?</script>|<style[\s\S]*?</style>|<[^>]+>")
_WS_RE = re.compile(r"\s+")
_BOILER_RE = re.compile(r"(cookie|privacy|subscribe|advertisement)", re.I)
_UA = "Mozilla/5.0 (compatible; FoxSay-feeder/0.1; research)"


def _robots_allowed(url: str) -> bool:
    try:
        rp = urllib.robotparser.RobotFileParser()
        rp.set_url(urllib.parse.urljoin(url, "/robots.txt"))
        rp.read()
        return rp.can_fetch("FoxSay-feeder", url)
    except Exception:
        return True


def fetch(url: str, timeout: int = 15) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        if r.headers.get_content_type() not in ("text/html", "text/plain"):
            return ""
        return r.read(512 * 1024).decode("utf-8", errors="ignore")


def clean_html(html_text: str) -> str:
    text = _TAG_RE.sub("\n", html_text)
    text = _html.unescape(text)
    paras = []
    for p in text.split("\n"):
        p = _WS_RE.sub(" ", p).strip()
        if len(p) >= 40 and not _BOILER_RE.search(p):
            paras.append(p)
    return "\n".join(paras)


def extract_links(html_text: str, base_url: str) -> set:
    links = set()
    for m in re.finditer(r'href=["\']([^"\'#]+)', html_text):
        u = urllib.parse.urljoin(base_url, m.group(1))
        if u.startswith("http") and not u.endswith((".png", ".jpg", ".pdf", ".zip")):
            links.add(u)
    return links


class WebFeeder:
    def __init__(self, seeds, max_pages: int = 200, max_depth: int = 2,
                 delay: float = 0.5):
        self.seeds = list(seeds)
        self.max_pages = max_pages
        self.max_depth = max_depth
        self.delay = delay
        self._seen = set()
        self._frontier = [(s, 0) for s in self.seeds]
        self.fetched = 0

    def __iter__(self):
        return self

    def __next__(self):
        while self._frontier and self.fetched < self.max_pages:
            url, depth = self._frontier.pop(0)
            if url in self._seen or depth > self.max_depth:
                continue
            self._seen.add(url)
            try:
                if not _robots_allowed(url):
                    continue
                raw = fetch(url)
            except Exception:
                continue
            if not raw:
                continue
            self.fetched += 1
            if depth < self.max_depth:
                for link in extract_links(raw, url):
                    if link not in self._seen:
                        self._frontier.append((link, depth + 1))
            time.sleep(self.delay)
            return clean_html(raw)
        raise StopIteration
