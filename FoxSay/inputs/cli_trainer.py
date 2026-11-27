"""
cli_trainer.py -- human-in-the-loop CLI chat training channel.

The human talks to the model in the terminal; every conversation turn is
immediately added to the training buffer, and the model fine-tunes on its own
conversation history right after each reply -- it literally grows while you
talk to it.  Also accepts direct teaching lines prefixed with 'teach:' that
skip generation and are trained on as-is.

Run:  python -m FoxSay.inputs.cli_trainer --shards_dir runs/x
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import torch
import torch.nn.functional as F

__all__ = ["CLIChatTrainer"]


class CLIChatTrainer:
    def __init__(self, model, tok, lr: float = 3e-4, max_new: int = 64,
                 temperature: float = 0.8, history_file: str = None,
                 device=None):
        self.model = model
        self.tok = tok
        self.lr = lr
        self.max_new = max_new
        self.temperature = temperature
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.opt = torch.optim.AdamW(model.parameters(), lr=lr)
        self.history_file = history_file
        self.history = []  # list of (speaker, text)

    def generate(self, prompt: str):
        ids = [i for i in self.tok.encode(prompt)[-256:]
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
        return self.tok.decode(x[0].tolist())[len(prompt):]

    def teach(self, text: str, steps: int = 1):
        """Train on one text immediately (used by chat turns and teach:)."""
        from .trainer import encode_batch
        x = encode_batch(self.tok, [text], self.model.config.vocab_size,
                         device=self.device)
        if x is None:
            return None
        xi, yi = x
        loss = self.model(xi, targets=yi)
        loss.backward()
        self.opt.step()
        self.opt.zero_grad()
        return float(loss)

    def run(self):
        print("CLI chat-trainer. 'teach: <text>' to teach directly, 'quit' to exit.")
        while True:
            try:
                line = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not line:
                continue
            if line.lower() in ("quit", "exit", "q"):
                break
            if line.lower().startswith("teach:"):
                text = line[6:].strip()
                loss = self.teach(text)
                print(f"taught ({loss if loss is not None else 'skip'})")
                self.history.append(("teacher", text))
                continue
            reply = self.generate(line)
            print(f"model> {reply}")
            self.history.append(("user", line))
            self.history.append(("model", reply))
            # immediate growth: train on the just-finished exchange
            exchange = f"user: {line}\nmodel: {reply}"
            loss = self.teach(exchange)
            if loss is not None:
                print(f"[learned loss={loss:.4f}]")
            if self.history_file:
                with open(self.history_file, "a", encoding="utf-8") as f:
                    f.write(exchange + "\n\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards_dir", default=None)
    ap.add_argument("--n_layer", type=int, default=4)
    ap.add_argument("--n_embd", type=int, default=256)
    ap.add_argument("--vocab_size", type=int, default=50257)
    cfg = ap.parse_args()

    import tiktoken
    tok = tiktoken.get_encoding("gpt2")

    from FoxSay.gpt import GPT, GPTConfig
    m = GPT(GPTConfig(n_layer=cfg.n_layer, n_embd=cfg.n_embd,
                      n_head=4, n_kv_head=4, vocab_size=cfg.vocab_size,
                      sequence_len=512))
    m.init_weights()
    if cfg.shards_dir:
        from FoxSay import brain_shards
        brain_shards.load_shards(m, cfg.shards_dir)
        print(f"[loaded] {cfg.shards_dir}")
    t = CLIChatTrainer(m, tok)
    t.run()


if __name__ == "__main__":
    main()
