"""
ebook_reader.py -- book / e-book parsing channel.

  .txt  -> read directly (streaming by chunks)
  .epub -> unzip + minimal XHTML chapter text extraction (stdlib only)
  .pdf  -> inject an extractor: EbookFeeder.set_pdf_extractor(fn)
           (stdlib cannot parse PDF; e.g. use pypdf if available)

Yields cleaned chapter texts, ready for the shared trainer.
"""

from __future__ import annotations

import os
import re
import zipfile
from xml.etree import ElementTree as ET

__all__ = ["EbookFeeder", "read_txt", "read_epub"]

_WS_RE = re.compile(r"\s+")


def read_txt(path: str, chunk_chars: int = 20000):
    with open(path, encoding="utf-8", errors="ignore") as f:
        buf = f.read()
    for i in range(0, len(buf), chunk_chars):
        yield _WS_RE.sub(" ", buf[i:i + chunk_chars]).strip()


def read_epub(path: str):
    """Yield chapter texts from an epub container (no external deps)."""
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        html_files = [n for n in names if n.endswith((".xhtml", ".html", ".htm"))
                      and not n.startswith("__MACOSX")]
        for name in html_files:
            try:
                raw = z.read(name).decode("utf-8", errors="ignore")
            except Exception:
                continue
            text = _strip_xml_tags(raw)
            text = _WS_RE.sub(" ", text).strip()
            if len(text) >= 80:
                yield text


def _strip_xml_tags(raw: str) -> str:
    # drop script/style first, then every tag; unescape basics
    raw = re.sub(r"<(script|style)[\s\S]*?</\1>", " ", raw, flags=re.I)
    raw = re.sub(r"<[^>]+>", " ", raw)
    raw = raw.replace("&nbsp;", " ").replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    return raw


class EbookFeeder:
    def __init__(self, paths):
        self.paths = list(paths)
        self._pdf_extractor = None

    def set_pdf_extractor(self, fn):
        """fn(path) -> str or iterable of str (e.g. a pypdf-based function)."""
        self._pdf_extractor = fn

    def __iter__(self):
        for path in self.paths:
            ext = os.path.splitext(path)[1].lower()
            try:
                if ext == ".txt":
                    yield from read_txt(path)
                elif ext == ".epub":
                    yield from read_epub(path)
                elif ext == ".pdf":
                    if self._pdf_extractor is None:
                        yield (f"[ebook] {os.path.basename(path)}: pdf needs "
                               f"set_pdf_extractor(fn) (install pypdf)")
                    else:
                        out = self._pdf_extractor(path)
                        if isinstance(out, str):
                            yield out
                        else:
                            yield from out
                else:
                    yield f"[ebook] unsupported type: {path}"
            except Exception as e:
                yield f"[ebook] {path}: read failed ({e})"
