"""
trainer.py -- shared streaming training loop for ALL input channels.

Feeds any iterable of text (or (prompt, text, logprobs) triples) into the
model one batch at a time; saves brain region shards every `save_every`
steps. This is the single place where every growth channel converges.
"""

from __future__ import annotations

import os

try:
    import torch
    import torch.nn.functional as F
    _HAS_TORCH = True
except ImportError:
    torch = None
    F = None
    _HAS_TORCH = False

__all__ = ["train_on_feeders", "encode_batch"]


def encode_batch(tok, texts, vocab_size, max_len=512, device="cpu"):
    """Encode a list of texts into (x, y) for next-token prediction."""
    xs, ys = [], []
    for t in texts:
        ids = [i for i in tok.encode(t)[:max_len] if i < vocab_size]
        if len(ids) < 2:
            continue
        if _HAS_TORCH and torch is not None:
            ids_t = torch.tensor(ids, device=device)
            xs.append(ids_t[:-1])
            ys.append(ids_t[1:])
        else:
            xs.append(ids[:-1])
            ys.append(ids[1:])
    if not xs:
        return None
    if _HAS_TORCH and torch is not None:
        x = torch.stack(xs)
        y = torch.stack(ys)
        return x, y
    return xs, ys


def train_on_feeders(model, tok, feeders, steps: int, out_dir: str,
                     n_shards: int = 3, save_every: int = 50, lr: float = 3e-4,
                     batch: int = 4, vocab_size: int = 50257,
                     loss_callback=None, device=None):
    """Stream texts from `feeders` (list of iterables, round-robin) and train.

    Each feeder may yield either:
      * str -> plain next-token SFT
      * (prompt, text, logprobs) -> distillation: KL against teacher logprobs
        (logprobs = (tokens, probs); None -> SFT on text)
    """
    if _HAS_TORCH and torch is not None:
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = device or "cpu"
    os.makedirs(out_dir, exist_ok=True)
    from FoxSay import brain_shards
    opt = torch.optim.AdamW(model.parameters(), lr=lr) if (_HAS_TORCH and torch is not None and hasattr(model, "parameters")) else None
    iters = [iter(f) for f in feeders]
    step = 0
    total_loss = 0.0
    while step < steps:
        items = []
        for it in iters:
            if len(items) >= batch:
                break
            try:
                items.append(next(it))
            except StopIteration:
                pass
        if not items:
            print("[trainer] all feeders exhausted")
            break
        loss = torch.zeros((), device=device) if (_HAS_TORCH and torch is not None) else 0.0
        n = 0
        for item in items:
            prompt, text, logp = None, None, None
            if isinstance(item, str):
                text = item
            else:
                prompt, text, logp = item
            if not text:
                text = prompt or ""
            if logp is not None:
                # distillation: KL on the teacher's tokens
                t_toks, t_probs = logp
                raw_ids = [
                    int(enc[0])
                    for enc in (tok.encode(t) for t in t_toks)
                    if enc and int(enc[0]) < int(vocab_size)
                ]
                if not raw_ids:
                    continue

                if _HAS_TORCH and torch is not None and F is not None:
                    x = encode_batch(tok, [text], vocab_size, device=device)
                    if x is None:
                        continue
                    xi, _ = x
                    logits = model(xi, targets=None)

                    max_len = min(int(xi.size(1)), int(logits.size(1)) - 1)
                    if max_len <= 0:
                        continue

                    t_idx = torch.tensor(raw_ids[:max_len], dtype=torch.long, device=device)
                    if t_idx.numel() == 0:
                        continue

                    n_tok = int(t_idx.numel())
                    t_probs_seq = [float(p) for p in t_probs[:n_tok]]
                    if not t_probs_seq:
                        continue

                    t_logp = torch.tensor(t_probs_seq, dtype=logits.dtype, device=device)
                    s_logp = F.log_softmax(logits[0, :n_tok], dim=-1)
                    picked = s_logp[torch.arange(n_tok, device=device), t_idx]
                    loss = loss + F.kl_div(picked, t_logp, reduction="sum")
                else:
                    # Fallback without torch: compute scalar KL divergence via NumPy / pure Python
                    logits_out = model(text) if callable(model) else None
                    if logits_out is None:
                        continue

                    try:
                        import numpy as np
                        arr = np.asarray(logits_out, dtype=np.float64)
                        if arr.ndim == 3:
                            step_logits = arr[0]
                        elif arr.ndim == 2:
                            step_logits = arr
                        else:
                            continue

                        max_len = max(0, min(len(raw_ids), step_logits.shape[0] - 1))
                        if max_len <= 0:
                            continue

                        idx_list = raw_ids[:max_len]
                        p_list = [float(p) for p in t_probs[:max_len]]
                        if not p_list:
                            continue

                        shifted = step_logits[:max_len] - np.max(step_logits[:max_len], axis=-1, keepdims=True)
                        log_probs = shifted - np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))
                        picked_np = log_probs[np.arange(max_len), np.asarray(idx_list, dtype=np.int64)]
                        target_np = np.asarray(p_list, dtype=np.float64)
                        kl_val = float(np.sum(target_np * (np.log(np.clip(target_np, 1e-12, None)) - picked_np)))
                    except ImportError:
                        import math
                        if not isinstance(logits_out, list) or not logits_out:
                            continue
                        rows = (
                            logits_out[0]
                            if (isinstance(logits_out[0], list) and logits_out[0] and isinstance(logits_out[0][0], (int, float)))
                            else logits_out
                        )
                        max_len = min(len(raw_ids), len(rows) - 1, len(t_probs))
                        if max_len <= 0:
                            continue
                        kl_val = 0.0
                        for pos in range(max_len):
                            row = rows[pos]
                            tid = raw_ids[pos]
                            tp = float(t_probs[pos])
                            if tid < 0 or tid >= len(row) or tp <= 0.0:
                                continue
                            m_val = max(row)
                            exps = [math.exp(v - m_val) for v in row]
                            z_val = sum(exps)
                            sp = exps[tid] / z_val if z_val > 0 else 1e-12
                            kl_val += tp * (math.log(max(tp, 1e-12)) - math.log(max(sp, 1e-12)))

                    loss = loss + kl_val
            else:
                x = encode_batch(tok, [text], vocab_size, device=device)
                if x is None:
                    continue
                xi, yi = x
                step_loss = model(xi, targets=yi)
                loss = loss + (step_loss if (_HAS_TORCH and torch is not None) else float(step_loss))
            n += 1
        if n == 0:
            continue
        if _HAS_TORCH and torch is not None and opt is not None and hasattr(loss, "backward"):
            (loss / n).backward()
            opt.step()
            opt.zero_grad()
            loss_val = float(loss.detach())
        else:
            loss_val = float(loss)
        step += 1
        total_loss += loss_val
        if loss_callback:
            loss_callback(step, loss_val / n)
        if step % 10 == 0:
            print(f"[trainer] step {step} loss {total_loss / 10:.4f}")
            total_loss = 0.0
        if step % save_every == 0:
            brain_shards.save_shards(model, out_dir, n_shards, step, optimizer=opt)
            print(f"[trainer] shards saved @ step {step}")
    brain_shards.save_shards(model, out_dir, n_shards, step, optimizer=opt)
    print(f"[trainer] done: {step} steps -> {out_dir}")
    return step
