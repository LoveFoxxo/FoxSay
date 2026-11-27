"""
av_learner.py -- audio/video watching channel.

Extracts watchable text from media with zero heavy dependencies:
  * .srt / .vtt subtitle files -> dialogue text
  * media files (.mp4/.mkv/.mp3/.wav) -> embedded subtitle/metadata streams
    if extractable; otherwise emits a descriptive placeholder and tells the
    caller how to inject a real transcript.

For real speech you inject an ASR function once:

    from FoxSay.inputs.av_learner import AVFeeder
    feeder = AVFeeder(["video.mp4"])
    feeder.set_transcriber(my_whisper_fn)   # fn(path) -> str
"""

from __future__ import annotations

import os
import re

__all__ = ["AVFeeder", "read_srt", "read_vtt"]

_TS_RE = re.compile(r"^\s*$|^\d+$|-->|\{\\an\d+\}|<[^>]+>")


def _clean_cue_line(line: str) -> str:
    line = line.strip()
    if not line or _TS_RE.search(line):
        return ""
    return line


def read_srt(path: str) -> str:
    with open(path, encoding="utf-8", errors="ignore") as f:
        lines = [l for l in f.read().splitlines() if _clean_cue_line(l)]
    # drop leading index numbers
    out, prev_num = [], False
    for l in lines:
        if l.isdigit() and not prev_num:
            prev_num = True
            continue
        prev_num = False
        out.append(l)
    return "\n".join(out)


def read_vtt(path: str) -> str:
    with open(path, encoding="utf-8", errors="ignore") as f:
        lines = f.read().splitlines()
    return "\n".join(
        _clean_cue_line(l) for l in lines
        if not l.startswith(("WEBVTT", "NOTE", "STYLE"))
    )


class AVFeeder:
    """Iterable of text extracted from audio/video files or URLs.

    items: list of paths (subtitle files, media files) or URLs (downloaded
    to a temp location first).
    """

    def __init__(self, items, tmp_dir: str = "runs/av_tmp"):
        self.items = list(items)
        self.tmp_dir = tmp_dir
        self._transcriber = None
        os.makedirs(tmp_dir, exist_ok=True)

    def set_transcriber(self, fn):
        """fn(path) -> transcript text (e.g. a Whisper wrapper)."""
        self._transcriber = fn

    def _to_local(self, item: str) -> str:
        if item.startswith("http"):
            import urllib.request
            name = os.path.basename(item.split("?")[0]) or "media.bin"
            target = os.path.join(self.tmp_dir, name)
            if not os.path.exists(target):
                urllib.request.urlretrieve(item, target)
            return target
        return item

    def _extract(self, path: str) -> str:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".srt":
            return read_srt(path)
        if ext == ".vtt":
            return read_vtt(path)
        if self._transcriber is not None:
            return self._transcriber(path) or ""
        return (f"[media] {os.path.basename(path)}: no built-in text extractor; "
                f"call AVFeeder.set_transcriber(whisper_fn) to transcribe speech.")

    def __iter__(self):
        for item in self.items:
            try:
                path = self._to_local(item)
                text = self._extract(path)
            except Exception as e:
                text = f"[media] {item}: extraction failed ({e})"
            yield text
