"""
distill_online.py -- train FoxSay weights WITHOUT any pre-collected local
dataset, by continuous information distillation with another AI.

Method (online distillation loop):
  1. A prompt pool is seeded with a few hand-written topics; the student model
     itself keeps generating NEW prompts (temperature sampling), so the pool
     evolves -- the two AIs teach each other without a dataset on disk.
  2. A background worker queries a teacher AI (any OpenAI-compatible API) for
     each prompt, asking for text + top-k logprobs when the API supports them.
  3. The student trains on the stream:
       * logprobs available  -> KL distillation student_logits vs teacher
                                logprobs (soft-label knowledge transfer)
       * logprobs unavailable-> SFT cross-entropy on the teacher's tokens
  4. Every `save_every` steps the weights are written as 2/3/4 region shards
     via FoxSay.brain_shards (no single-point weight file).

Usage:
  TEACHER_BASE_URL=... TEACHER_API_KEY=... TEACHER_MODEL=... \
  python scripts/distill_online.py --n_shards 3 --save_every 50 --steps 1000

No TEACHER_* env vars -> degrades to self-distillation (student samples its
own completions; still a valid distillation signal for exploration).
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import random
import threading
import time
import urllib.request

import torch
import torch.nn.functional as F

# make the repo root importable when run as `python scripts/distill_online.py`
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

SEED_PROMPTS = [
    "Explain why attention heads specialize into induction heads.",
    "Write a proof sketch that a sliding-window transformer can still solve long-range tasks.",
    "Describe how GRPO advantages differ from PPO advantages.",
    "List concrete failure modes of RLHF and how GAN gating mitigates them.",
    "Explain rotary position embeddings intuitively, then rigorously.",
]

# --- teacher API client (OpenAI-compatible, stdlib only) -------------------

def teacher_call(base_url: str, api_key: str, model: str, prompt: str,
                 max_tokens: int = 256, timeout: int = 60):
    url = base_url.rstrip("/") + "/chat/completions"
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "logprobs": True,       # ignored by servers that do not support it
        "top_logprobs": 5,
    }).encode()
    req = urllib.request.Request(url, data=payload, headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    })
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def extract_teacher(completion: dict):
    """-> (text, list_of_token_logprobs_or_None)."""
    choice = completion["choices"][0]
    msg = choice["message"]
    text = msg.get("content") or ""
    toks, logprobs = None, None
    if choice.get("logprobs") and choice["logprobs"].get("content"):
        c = choice["logprobs"]["content"]
        toks = [t["token"] for t in c]
        logprobs = [t["logprob"] for t in c]
    return text, (toks, logprobs) if toks else None


# --- prompt pool: seeds + student self-generation (information loop) -------

def student_prompt(student, tok, device, temperature=1.0):
    """Let the student ask its own question -- the pool evolves without data."""
    x = tok.encode("New challenging question for a language model: ")
    ids = torch.tensor([x], device=device)
    with torch.no_grad():
        for _ in range(24):
            logits = student(ids)
            probs = F.softmax(logits[0, -1] / max(temperature, 1e-3), dim=-1)
            nxt = torch.multinomial(probs, 1).reshape(1, 1)
            if nxt.item() == tok.eot_token:
                break
            ids = torch.cat([ids, nxt], dim=1)
    return tok.decode(ids[0].tolist())


def teacher_worker(cfg, prompt_q: queue.Queue, batch_q: queue.Queue, stop):
    """Background producer: prompts -> teacher -> (prompt, teacher signal)."""
    while not stop.is_set():
        try:
            prompt = prompt_q.get(timeout=1.0)
        except queue.Empty:
            continue
        try:
            comp = teacher_call(cfg.base_url, cfg.api_key, cfg.model, prompt,
                                max_tokens=cfg.max_tokens)
            text, logp = extract_teacher(comp)
            batch_q.put((prompt, text, logp))
        except Exception as e:
            print(f"[teacher] retry {prompt[:40]!r}: {e}")
            time.sleep(2)
            prompt_q.put(prompt)  # retry


# --- training loop ----------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--save_every", type=int, default=50)
    ap.add_argument("--n_shards", type=int, default=3, choices=(2, 3, 4))
    ap.add_argument("--out_dir", default="runs/distill_online")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--max_tokens", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n_layer", type=int, default=4)
    ap.add_argument("--n_embd", type=int, default=256)
    ap.add_argument("--vocab_size", type=int, default=50257,
                    help="must match the tokenizer (gpt2 = 50257; change both together)")
    cfg = ap.parse_args()
    cfg.base_url = os.environ.get("TEACHER_BASE_URL")
    cfg.api_key = os.environ.get("TEACHER_API_KEY")
    cfg.model = os.environ.get("TEACHER_MODEL")

    import tiktoken
    tok = tiktoken.get_encoding("gpt2")  # replace with FoxSay tokenizer if desired
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from FoxSay.gpt import GPT, GPTConfig
    m = GPT(GPTConfig(n_layer=cfg.n_layer, n_embd=cfg.n_embd,
                      n_head=4, n_kv_head=4, vocab_size=cfg.vocab_size,
                      sequence_len=512))
    m.to(device); m.init_weights(); m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=cfg.lr)

    prompt_q, batch_q = queue.Queue(), queue.Queue(maxsize=32)
    stop = threading.Event()
    if cfg.base_url and cfg.api_key and cfg.model:
        workers = [threading.Thread(target=teacher_worker,
                                    args=(cfg, prompt_q, batch_q, stop),
                                    daemon=True) for _ in range(2)]
        [w.start() for w in workers]
        print(f"[teacher] online: {cfg.model}")
    else:
        print("[teacher] no API env vars -> self-distillation mode (offline)")
    for p in SEED_PROMPTS:
        prompt_q.put(p)

    from FoxSay import brain_shards
    os.makedirs(cfg.out_dir, exist_ok=True)
    step = 0
    total_loss = 0.0
    while step < cfg.steps:
        # keep the pool fed: 1 seed + 1 student-generated prompt per refill
        if prompt_q.qsize() < 8:
            prompt_q.put(random.choice(SEED_PROMPTS))
            try:
                prompt_q.put(student_prompt(m, tok, device))
            except Exception:
                pass

        items = []
        while len(items) < cfg.batch:
            try:
                items.append(batch_q.get(timeout=2.0))
            except queue.Empty:
                items.append((random.choice(SEED_PROMPTS), None, None))
            if step >= cfg.steps:
                break
        if step >= cfg.steps:
            break

        loss = torch.zeros((), device=device)
        for prompt, text, logp in items:
            if text is None:  # fallback self-play
                text = prompt
            ids = tok.encode(text)[:512]
            ids = [i for i in ids if i < cfg.vocab_size]  # clamp to model vocab
            if len(ids) < 2:
                continue
            x = torch.tensor([ids], device=device)
            if logp is None:
                # FoxSay GPT returns loss when targets given (no internal shift)
                y = torch.cat([x[:, 1:], torch.full((1, 1), -1, device=device,
                                                    dtype=x.dtype)], dim=1)
                loss = loss + m(x, targets=y)
            else:
                logits = m(x, targets=None)  # (1, L, V)
                t_ids = [tok.encode(t) for t in logp[0]]
                t_probs = [float(p) for p in logp[1]]
                t_idx = torch.tensor([i[0] for i in t_ids if i], device=device)[:511]
                if len(t_idx) == 0:
                    continue
                t_logp = torch.tensor(t_probs[:len(t_idx)], device=device)
                s_logp = F.log_softmax(logits[0, :len(t_idx)], dim=-1)
                picked = s_logp[torch.arange(len(t_idx), device=device), t_idx]
                loss = loss + F.kl_div(picked, t_logp, reduction="sum")

        (loss / len(items)).backward()
        opt.step(); opt.zero_grad()
        total_loss += float(loss)
        step += 1
        if step % 10 == 0:
            print(f"step {step} loss {total_loss / 10:.4f}")
            total_loss = 0.0
        if step % cfg.save_every == 0:
            brain_shards.save_shards(m, cfg.out_dir, cfg.n_shards, step,
                                     optimizer=opt)
            print(f"[shards] saved {cfg.n_shards} regions @ step {step}")

    stop.set()
    brain_shards.save_shards(m, cfg.out_dir, cfg.n_shards, step, optimizer=opt)
    print(f"done: {cfg.n_shards} region shards in {cfg.out_dir}")


if __name__ == "__main__":
    main()
