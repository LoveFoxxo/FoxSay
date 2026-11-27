"""
media_inbox.py -- incoming media (images / videos / documents) reception.

MediaInbox stores uploads under a save dir and appends a JSONL index entry
per item.  It doubles as a feeder for FoxSay.inputs: the stored items can
be handed to AVFeeder (video/audio), EbookFeeder (docs/books), or the
modality router as external signals.

Message format (shared with web_api):
    {"role": "external", "kind": "image"|"video"|"document"|"other",
     "content": str, "path": str, "ts": float}
"""

from __future__ import annotations

import json
import os
import time

__all__ = ["MediaInbox"]

KIND_MAP = {
    ".png": "image", ".jpg": "image", ".jpeg": "image", ".gif": "image",
    ".webp": "image", ".bmp": "image",
    ".mp4": "video", ".mkv": "video", ".mov": "video", ".avi": "video",
    ".webm": "video",
    ".mp3": "audio", ".wav": "audio", ".ogg": "audio", ".flac": "audio",
    ".txt": "document", ".pdf": "document", ".epub": "document",
    ".docx": "document", ".md": "document", ".html": "document",
}


class MediaInbox:
    def __init__(self, save_dir: str, index_file: str = None):
        self.save_dir = save_dir
        self.index_file = index_file or os.path.join(save_dir, "inbox.jsonl")
        os.makedirs(save_dir, exist_ok=True)

    def receive(self, filename: str, data: bytes, description: str = "") -> dict:
        """Store one upload, return its message record."""
        safe = os.path.basename(filename) or "upload.bin"
        path = os.path.join(self.save_dir, safe)
        # never overwrite silently: unique suffix
        n = 1
        while os.path.exists(path):
            stem, ext = os.path.splitext(safe)
            path = os.path.join(self.save_dir, f"{stem}_{n}{ext}")
            n += 1
        with open(path, "wb") as f:
            f.write(data)
        kind = KIND_MAP.get(os.path.splitext(safe)[1].lower(), "other")
        msg = {
            "role": "external", "kind": kind,
            "content": description, "path": path, "ts": time.time(),
        }
        with open(self.index_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(msg, ensure_ascii=False) + "\n")
        return msg

    def items(self):
        """All records received so far."""
        if not os.path.exists(self.index_file):
            return []
        out = []
        with open(self.index_file, encoding="utf-8") as f:
            for line in f:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return out

    def by_kind(self, kind: str):
        return [m for m in self.items() if m["kind"] == kind]

    def as_feeder(self):
        """Yield item descriptions (content + path) for the inputs package."""
        for m in self.items():
            yield f"[{m['kind']}] {m['content']} -> {m['path']}"
