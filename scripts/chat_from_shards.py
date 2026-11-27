"""
chat_from_shards.py -- load a model from brain region shards and chat with it.

Usage:
  python scripts/chat_from_shards.py --shards_dir runs/dry_distill
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F


def generate(model, tok, prompt: str, max_new: int, temperature: float, device):
    ids = tok.encode(prompt)[-256:]  # keep within rotary cache headroom
    while len(ids) < 2:              # GPT.forward asserts T > 1
        ids.append(tok.eot_token)
    x = torch.tensor([ids], device=device)
    model.eval()
    with torch.no_grad():
        for _ in range(max_new):
            logits = model(x, targets=None)          # (1, T, V)
            probs = F.softmax(logits[0, -1] / max(temperature, 1e-3), dim=-1)
            nxt = torch.multinomial(probs, 1).reshape(1, 1)
            if nxt.item() == tok.eot_token:
                break
            x = torch.cat([x, nxt], dim=1)
            if x.size(1) >= 500:                      # rotary cache limit guard
                break
    out = tok.decode(x[0].tolist())
    return out[len(prompt):] if out.startswith(prompt) else out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards_dir", default="runs/dry_distill")
    ap.add_argument("--n_layer", type=int, default=2)
    ap.add_argument("--n_embd", type=int, default=64)
    ap.add_argument("--vocab_size", type=int, default=50257)
    ap.add_argument("--max_new", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.8)
    cfg = ap.parse_args()

    import tiktoken
    tok = tiktoken.get_encoding("gpt2")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from FoxSay.gpt import GPT, GPTConfig
    m = GPT(GPTConfig(n_layer=cfg.n_layer, n_embd=cfg.n_embd,
                      n_head=4, n_kv_head=4, vocab_size=cfg.vocab_size,
                      sequence_len=512))
    m.init_weights(); m.to(device)

    from FoxSay import brain_shards
    res = brain_shards.load_shards(m, cfg.shards_dir)
    ok = [r for r, v in res.items() if v]
    print(f"[loaded] regions {ok} from {cfg.shards_dir}")

    print("Chat mode. Type 'quit' to exit.")
    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue
        if line.lower() in ("quit", "exit", "q"):
            break
        reply = generate(m, tok, line, cfg.max_new, cfg.temperature, device)
        print(f"model> {reply}")


if __name__ == "__main__":
    main()
