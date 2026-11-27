"""
teacher_api.py -- multi-AI conversation distillation channel.

TeacherStream yields (prompt, text, logprobs) triples by calling any
OpenAI-compatible chat API.  The prompt source can be any iterable
(seed prompts, the student's own questions, another AI's feed), so the
distillation loop keeps generating its own training signal -- no dataset.

Set TEACHER_BASE_URL / TEACHER_API_KEY / TEACHER_MODEL, or pass them in.
"""

from __future__ import annotations

import json
import os
import time
import urllib.request

__all__ = ["TeacherStream"]


def _call(base_url: str, api_key: str, model: str, prompt: str,
          max_tokens: int, timeout: int):
    url = base_url.rstrip("/") + "/chat/completions"
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "logprobs": True,
        "top_logprobs": 5,
    }).encode()
    req = urllib.request.Request(url, data=payload, headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    })
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _extract(completion):
    choice = completion["choices"][0]
    msg = choice["message"]
    text = msg.get("content") or ""
    toks = logprobs = None
    if choice.get("logprobs") and choice["logprobs"].get("content"):
        c = choice["logprobs"]["content"]
        toks = [t["token"] for t in c]
        logprobs = [t["logprob"] for t in c]
    return text, ((toks, logprobs) if toks else None)


class TeacherStream:
    """Iterable of (prompt, text, logprobs) from a teacher AI.

    prompts: iterable of str (may be infinite, e.g. the student's own
             self-generated questions).
    """

    def __init__(self, prompts, base_url=None, api_key=None, model=None,
                 max_tokens: int = 256, retries: int = 3):
        self.prompts = iter(prompts)
        self.base_url = base_url or os.environ.get("TEACHER_BASE_URL")
        self.api_key = api_key or os.environ.get("TEACHER_API_KEY")
        self.model = model or os.environ.get("TEACHER_MODEL")
        self.max_tokens = max_tokens
        self.retries = retries

    @property
    def online(self):
        return bool(self.base_url and self.api_key and self.model)

    def __iter__(self):
        return self

    def __next__(self):
        prompt = next(self.prompts)
        if not self.online:
            # offline degradation: echo the prompt (self-distillation SFT)
            return prompt, prompt, None
        for attempt in range(self.retries):
            try:
                comp = _call(self.base_url, self.api_key, self.model,
                             prompt, self.max_tokens, timeout=60)
                text, logp = _extract(comp)
                return prompt, text, logp
            except Exception as e:
                if attempt == self.retries - 1:
                    print(f"[teacher] giving up {prompt[:40]!r}: {e}")
                    return prompt, prompt, None
                time.sleep(2 * (attempt + 1))
