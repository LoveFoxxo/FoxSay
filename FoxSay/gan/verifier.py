"""
Generative-adversarial verifier ("GAN" style) for the internal verification loop.

Roles
=====
1) Discriminator / Process Reward Model (PRM). A small transformer that reads
   the current turn (user question + candidate assistant answer, already
   tokenized) and emits
       - a per-chunk "coherence/quality" score  (process reward, PRM-style)
       - an aggregated pass/fair logit for the WHOLE turn (discriminator head)
   The same network plays both roles, exactly like a value function and a
   critic can share a trunk: chunk scores are the "feature critic", the turn
   logit is the "outcome critic".

2) Adversarial training target. The policy (the big GPT) wants high verifier
   scores; the verifier itself is trained to separate
       positives: ground-truth (or reward==1) answers
       negatives: the policy's own wrong/hallucinated answers
   This is a classic minimax setup (GAN-family, LSGAN loss for stability) -
   and it is exactly how modern "generative adversarial PRM" pipelines shape
   RL rewards. It never touches the policy weights directly: it only shapes
   the reward the policy sees, so it cannot destabilize the policy gradient.

3) Inference-time output gate. At the moment the policy samples
   <|assistant_end|>, the candidate answer is scored by the verifier; the
   turn is only "released" (accepted as final output) if
       P(pass) >= `gate_threshold`
   otherwise the end-token is masked and the policy must keep thinking
   (the loop budget is increased, letting latent cycles refine further).
   After `max_rejects` rejections the turn is released anyway (no deadlock).

Everything is CPU-friendly and FSDP/FP64-friendly: plain nn.Linear/Embedding
created outside meta-device, params go through the normal MuonAdamW/AdamW
optimizer path when `attach_to_optimizer_groups()` is used.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GPTConfigLite:
    """Minimal config for the verifier (intentionally NOT the main GPTConfig)."""

    def __init__(self, n_embd=256, n_layer=4, n_head=4, max_chunk_tokens=64, vocab_size=32768):
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.max_chunk_tokens = max_chunk_tokens
        self.vocab_size = vocab_size


class VerifierBlock(nn.Module):
    """One pre-norm transformer block of the verifier (rope-free, causal)."""

    def __init__(self, cfg):
        super().__init__()
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        assert cfg.n_embd % cfg.n_head == 0
        self.c_qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=False)
        self.c_fc2 = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=False)
        self.pos = nn.Embedding(8192, cfg.n_embd)

    def forward(self, x, t0):
        B, T, C = x.shape
        pos_ids = (t0 + torch.arange(T, device=x.device)).clamp(max=self.pos.num_embeddings - 1)
        x = x + self.pos(pos_ids).unsqueeze(0)
        qkv = self.c_qkv(F.rms_norm(x, (C,))).view(B, T, 3, self.n_head, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]  # (B, T, H, D)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))    # (B, H, T, D)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = self.c_proj(y.transpose(1, 2).reshape(B, T, C))
        x = x + y
        h = self.c_fc2(F.relu(self.c_fc(F.rms_norm(x, (C,)))) ** 2)
        x = x + h
        return x


class VerifierModel(nn.Module):
    """Discriminator + process-reward model over one dialogue turn.

    Input is the FULL turn token stream (question + candidate answer, special
    tokens included). Output:
      chunk_scores: (B, n_chunks) quality score per ~max_chunk_tokens chunk
      turn_logit:   (B,)   positive = looks like a good, complete answer
    """

    def __init__(self, config: GPTConfigLite = None):
        super().__init__()
        self.config = config or GPTConfigLite()
        c = self.config
        self.wte = nn.Embedding(c.vocab_size, c.n_embd)
        self.blocks = nn.ModuleList([VerifierBlock(c) for _ in range(c.n_layer)])
        self.chunk_head = nn.Linear(c.n_embd, 1)   # per-chunk score head
        self.turn_head = nn.Linear(c.n_embd, 1)    # whole-turn discriminator head

    @torch.no_grad()
    def init_weights(self):
        c = self.config
        torch.nn.init.normal_(self.wte.weight, std=0.02)
        s = 3 ** 0.5 * c.n_embd ** -0.5
        for b in self.blocks:
            torch.nn.init.uniform_(b.c_qkv.weight, -s, s)
            torch.nn.init.zeros_(b.c_proj.weight)
            torch.nn.init.uniform_(b.c_fc.weight, -s * 0.4, s * 0.4)
            torch.nn.init.zeros_(b.c_fc2.weight)
            torch.nn.init.normal_(b.pos.weight, std=0.02)
        torch.nn.init.zeros_(self.chunk_head.weight)
        torch.nn.init.zeros_(self.chunk_head.bias)
        # turn head starts slightly negative => untrained gate REJECTS nothing
        # accepted by default? No: untrained score ~ 0 => we bias acceptance
        # in the gate itself (see OutputGate). Keep the head at zero so the
        # first training step defines the direction from data, not from init.
        torch.nn.init.zeros_(self.turn_head.weight)
        torch.nn.init.zeros_(self.turn_head.bias)

    def forward(self, tokens, chunk_boundaries=None):
        """
        tokens: (B, T) int64 - the full turn (question + candidate answer).
        chunk_boundaries: optional list of (B,) lists of chunk end positions.
            If None, fixed-size chunks of max_chunk_tokens are used.
        Returns (chunk_scores (B, K), turn_logit (B,)).
        """
        B, T = tokens.shape
        x = self.wte(tokens)
        for blk in self.blocks:
            x = blk(x, t0=0)
        # per-token scores -> per-chunk scores
        tok_score = self.chunk_head(F.rms_norm(x, (x.size(-1),))).squeeze(-1)  # (B, T)
        if chunk_boundaries is None:
            c = self.config.max_chunk_tokens
            boundaries = [[min((k + 1) * c, T) for k in range((T + c - 1) // c)] for _ in range(B)]
        else:
            boundaries = chunk_boundaries
        max_chunks = max(len(b) for b in boundaries)
        chunk_scores = torch.zeros(B, max_chunks, device=x.device, dtype=tok_score.dtype)
        for b in range(B):
            start = 0
            for j, end in enumerate(boundaries[b]):
                # mean over the chunk; empty chunk -> 0
                if end > start:
                    chunk_scores[b, j] = tok_score[b, start:end].mean()
                start = end
        # turn logit from mean-pooled final state
        turn_logit = self.turn_head(x.mean(dim=1)).squeeze(-1)
        return chunk_scores, turn_logit

    def process_reward(self, tokens, chunk_boundaries=None, normalize=True):
        """PRM-style dense reward: mean chunk score, tanh-bounded to (-1, 1)."""
        chunk_scores, _ = self(tokens, chunk_boundaries)
        r = chunk_scores.mean(dim=1)
        if normalize:
            r = torch.tanh(r)
        return r


class VerifierLoss(nn.Module):
    """LSGAN-style minimax loss between the verifier and the policy answers.

    D tries:  score(positive) -> +1,  score(negative) -> -1   (hinge-free L2)
    Optionally includes a rank consistency term (positive > negative).
    """

    def __init__(self, rank_margin=0.5, rank_weight=0.5):
        super().__init__()
        self.rank_margin = rank_margin
        self.rank_weight = rank_weight

    def forward(self, pos_logit, neg_logit):
        """pos_logit/neg_logit: (N,) turn logits for correct / wrong answers."""
        if pos_logit.numel() == 0 or neg_logit.numel() == 0:
            # one side empty -> pure L2 on the available side, no rank term
            if pos_logit.numel() > 0:
                return ((pos_logit - 1.0) ** 2).mean(), torch.tensor(0.0)
            if neg_logit.numel() > 0:
                return ((neg_logit + 1.0) ** 2).mean(), torch.tensor(0.0)
            zero = pos_logit.new_zeros(())
            return zero, zero.clone()
            return pos_logit.sum() * 0.0, torch.tensor(0.0)
        l_pos = ((pos_logit - 1.0) ** 2).mean()
        l_neg = ((neg_logit + 1.0) ** 2).mean()
        rank = F.relu(self.rank_margin - (pos_logit.unsqueeze(1) - neg_logit.unsqueeze(0))).mean()
        return l_pos + l_neg, rank

    def total(self, pos_logit, neg_logit):
        l, rank = self.forward(pos_logit, neg_logit)
        return l + self.rank_weight * rank


class OutputGate:
    """Inference-time final-output threshold gate.

    DECISION RULE (the "door" for releasing the policy's answer):
      accept  <=>  sigmoid(turn_logit) >= gate_threshold
      reject  =>   mask <|assistant_end|>, let the policy keep thinking
                   (and the latent loop gets one more cycle of budget);
                   after max_rejects rejections, release anyway (no deadlock).
    """

    def __init__(self, verifier: VerifierModel, gate_threshold=0.5, max_rejects=3):
        self.verifier = verifier
        self.gate_threshold = gate_threshold
        self.max_rejects = max_rejects

    @torch.no_grad()
    def decide(self, tokens_batch, chunk_boundaries=None):
        """tokens_batch: (B, T) full-turn tensors. Returns (accept (B,) bool, p (B,))."""
        _, turn_logit = self.verifier(tokens_batch, chunk_boundaries)
        p = torch.sigmoid(turn_logit)
        return p >= self.gate_threshold, p


class VerifierTrainer:
    """Adversarial trainer for the verifier.

    Positive pool: ground-truth / reward==1 samples.
    Negative pool: wrong policy samples (and hard negatives: right answer,
    broken reasoning, if the caller can label them).
    The verifier is trained with AdamW (LSGAN objective); the policy is NOT
    updated here - its pressure comes through the shaped reward.
    """

    def __init__(self, verifier: VerifierModel, lr=1e-4, weight_decay=0.01, device=None):
        self.verifier = verifier
        self.device = device or next(verifier.parameters()).device
        self.verifier.to(self.device)
        self.optimizer = torch.optim.AdamW(
            [p for p in self.verifier.parameters() if p.requires_grad],
            lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95),
        )
        self.loss_fn = VerifierLoss()

    def train_batch(self, pos_tokens, neg_tokens):
        """pos_tokens / neg_tokens: (N, T) padded token tensors (can differ in T).
        Returns (total_loss, rank_loss) detached floats."""
        self.verifier.train()
        self.optimizer.zero_grad(set_to_none=True)
        losses, ranks = [], []
        for tokens in (pos_tokens, neg_tokens):
            if tokens is not None and tokens.numel() > 0:
                _, logit = self.verifier(tokens.to(self.device))
                losses.append(logit)
        pos_logit = losses[0] if len(losses) > 0 else torch.zeros(0, device=self.device)
        neg_logit = losses[1] if len(losses) > 1 else torch.zeros(0, device=self.device)
        loss, rank = self.loss_fn(pos_logit, neg_logit)
        total = loss + self.loss_fn.rank_weight * rank
        total.backward()
        torch.nn.utils.clip_grad_norm_(self.verifier.parameters(), 1.0)
        self.optimizer.step()
        return total.item(), rank.item()

    @torch.no_grad()
    def evaluate(self, tokens):
        if tokens is None or tokens.numel() == 0:
            return 0.0
        self.verifier.eval()
        _, logit = self.verifier(tokens.to(self.device))
        return torch.sigmoid(logit).mean().item()


def collate_turns(turn_lists, pad_id, device=None):
    """Pad a list of variable-length token lists into (N, T) LongTensor."""
    max_len = max(len(t) for t in turn_lists)
    out = torch.full((len(turn_lists), max_len), pad_id, dtype=torch.long)
    for i, t in enumerate(turn_lists):
        out[i, :len(t)] = torch.as_tensor(t, dtype=torch.long)
    if device is not None:
        out = out.to(device)
    return out
