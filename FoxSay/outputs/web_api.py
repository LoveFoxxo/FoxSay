"""
web_api.py -- the model's web portal: text in/out + media upload, stdlib only.

Endpoints:
  POST /message   {"text": "..."}              -> {"reply": "..."}
  GET  /history                                 -> last N exchanges
  POST /upload    (raw bytes; ?name=file.png)   -> MediaInbox.receive
  GET  /items     (query by kind)               -> MediaInbox.items
  GET  /          tiny status page

Usage:
  from FoxSay.outputs import WebPortal
  portal = WebPortal(model, tok, port=8080)
  portal.serve_forever()
"""

from __future__ import annotations

import json
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
import torch.nn.functional as F

from .media_inbox import MediaInbox

__all__ = ["WebPortal"]


class WebPortal:
    def __init__(self, model, tok, port: int = 8080, save_dir: str = "runs/inbox",
                 max_new: int = 128, temperature: float = 0.8, device=None,
                 max_history: int = 100):
        self.model = model
        self.tok = tok
        self.port = port
        self.max_new = max_new
        self.temperature = temperature
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.inbox = MediaInbox(save_dir)
        self.history = []
        self.max_history = max_history

    def generate(self, text: str) -> str:
        ids = [i for i in self.tok.encode(text)[-256:]
               if i < self.model.config.vocab_size]
        if not ids:
            return ""
        while len(ids) < 2:  # GPT.forward asserts T > 1
            ids.append(self.tok.eot_token)
        x = torch.tensor([ids], device=self.device)
        self.model.eval()
        with torch.no_grad():
            for _ in range(self.max_new):
                logits = self.model(x, targets=None)
                probs = F.softmax(logits[0, -1] / max(self.temperature, 1e-3), dim=-1)
                nxt = torch.multinomial(probs, 1).reshape(1, 1)
                if nxt.item() == self.tok.eot_token:
                    break
                x = torch.cat([x, nxt], dim=1)
                if x.size(1) >= 500:
                    break
        self.model.train()
        return self.tok.decode(x[0].tolist())[len(text):]

    def reply(self, text: str) -> str:
        r = self.generate(text)
        self.history.append({"role": "user", "kind": "text", "content": text,
                             "path": None, "ts": time.time()})
        self.history.append({"role": "model", "kind": "text", "content": r,
                             "path": None, "ts": time.time()})
        self.history = self.history[-self.max_history:]
        return r

    # -- http plumbing -------------------------------------------------------
    def _make_handler(self):
        portal = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass  # quiet

            def _send(self, code: int, obj):
                body = json.dumps(obj, ensure_ascii=False).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                path = urllib.parse.urlparse(self.path).path
                if path == "/":
                    self._send(200, {"status": "FoxSay portal",
                                     "endpoints": ["/message", "/history", "/upload", "/items"]})
                elif path == "/history":
                    self._send(200, {"history": portal.history})
                elif path == "/items":
                    q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
                    kind = q.get("kind", [None])[0]
                    items = portal.inbox.items()
                    if kind:
                        items = [m for m in items if m["kind"] == kind]
                    self._send(200, {"items": items})
                else:
                    self._send(404, {"error": "not found"})

            def do_POST(self):
                path = urllib.parse.urlparse(self.path).path
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length) if length else b""
                if path == "/message":
                    try:
                        text = json.loads(body).get("text", "")
                        self._send(200, {"reply": portal.reply(text)})
                    except Exception as e:
                        self._send(400, {"error": str(e)})
                elif path == "/upload":
                    q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
                    name = q.get("name", ["upload.bin"])[0]
                    desc = q.get("desc", [""])[0]
                    msg = portal.inbox.receive(name, body, desc)
                    self._send(200, msg)
                else:
                    self._send(404, {"error": "not found"})

        return Handler

    def serve_forever(self):
        srv = ThreadingHTTPServer(("0.0.0.0", self.port), self._make_handler())
        print(f"[web_api] serving on http://0.0.0.0:{self.port}")
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            srv.shutdown()

    def start_background(self):
        t = threading.Thread(target=self.serve_forever, daemon=True)
        t.start()
        return t
