"""
GPT model (rewrite, a lot simpler)
Notable features:
- rotary embeddings (and no positional embeddings)
- QK norm
- untied weights for token embedding and lm_head
- relu^2 activation in MLP
- norm after token embedding
- no learnable params in rmsnorm
- no bias in linear layers
- Group-Query Attention (GQA) support for more efficient inference
- Flash Attention 3 integration
"""

from functools import partial
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from FoxSay.common import get_dist_info, print0, COMPUTE_DTYPE
from FoxSay.optim import MuonAdamW
from FoxSay.gan import LoopGate, neutral_state as loop_neutral_state
from FoxSay.moe import SparseMoE
from FoxSay.numerics import rms_norm

# Our custom Flash Attention module that automatically uses FA3 when compatible and SDPA fallback otherwise
from FoxSay.flash_attention import flash_attn

@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6 # number of query heads
    n_kv_head: int = 6 # number of key/value heads (GQA)
    n_embd: int = 768
    # Sliding window attention pattern string, tiled across layers. Final layer always L.
    # Characters: L=long (full context), S=short (quarter context)
    # Examples: "L"=all full context, "SL"=alternating, "SSL"=two short then one long
    window_pattern: str = "SSSL"
    # Sparse expansion: zero experts preserves the stock architecture/checkpoints.
    moe_num_experts: int = 0
    moe_expert_ratio: float = 4.0
    moe_top_k: int = 1
    moe_every: int = 1
    moe_aux_loss_scale: float = 0.01
    moe_router_z_loss_scale: float = 0.001
    # Numerical floor controls. loss_dtype may be float32 or float64.
    loss_chunk_size: int = 0
    loss_dtype: str = "float32"
    logit_softcap: float = 15.0  # <= 0 disables saturation for deterministic tasks
    # --- Hierarchical sliding window (SuperWindow + modality sub-windows) ---
    # When True, each layer's static window is specialized per modality focus:
    # a tiny router pools hidden states, four sub-windows (text/audio/image/
    # video) each propose a window, and the main window aggregates them into
    # one unified (left, right). The static pattern stays the hard cap.
    modality_window: bool = False
    modality_sub_factors: dict = None   # override per-modality left-window factor
    window_smooth: float = 0.9          # EMA factor for unified window updates
    # --- Sub-GAN cluster (2/3/4 depth-mounted verifiers, see gan/subgans.py) ---
    # None = off (legacy bit-identical). Else depth fractions of the trunk,
    # e.g. [0.5, 0.75] mounts 2 sub-GANs at half and 3/4 depth.
    subgan_layers: list = None
    # --- Mind: the mathematical substrate (see FoxSay/mind) ---
    # True enables TD-critic emotion, manifold thought sampling, spectral
    # ensemble, Hopfield memory etc. inside the internal verification loop.
    mind: bool = False
    # --- Internal verification loop (latent recurrence, see FoxSay/gan) ---
    # loop_max_cycles=1 disables the loop entirely (bit-identical legacy model).
    # loop_max_cycles>1 enables ACT-style latent cycles: cycle 1 commits the
    # K/V projections; later cycles refine the stream while attention stays
    # anchored to the committed K/V. Zero-initialized gate/mix params mean an
    # untrained loop is bit-identical to the legacy forward pass.
    loop_max_cycles: int = 1
    # Hard halting threshold at inference: stop refining once p_stop >= this.
    loop_stop_threshold: float = 0.5


def norm(x):
    return rms_norm(x)

class Linear(nn.Linear):
    """nn.Linear that casts weights to match input dtype in forward.
    Replaces autocast: master weights stay fp32 for optimizer precision,
    but matmuls run in the activation dtype (typically bf16 from embeddings).
    The bias (when present, fp32/fp64 master) is cast the same way so the
    add never hits a dtype mismatch."""
    def forward(self, x):
        w = self.weight.to(dtype=x.dtype)
        b = None if self.bias is None else self.bias.to(dtype=x.dtype)
        return F.linear(x, w, b)


def has_ve(layer_idx, n_layer):
    """Returns True if GPT layer should have Value Embedding (alternating, last layer always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2

def apply_rotary_emb(x, cos, sin):
    # note: this rotates by -theta, the transpose of the textbook convention. Functionally
    # equivalent (only the relative q/k rotation matters), kept for checkpoint compatibility.
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
    y1 = x1 * cos + x2 * sin # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)

class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 12
        # bias=True: the gate needs a baseline logit (RMSNorm does not remove
        # means, so the gate bias absorbs the activation's DC offset)
        self.ve_gate = Linear(self.ve_gate_channels, self.n_kv_head, bias=True) if has_ve(layer_idx, config.n_layer) else None

    def forward(self, x, ve, cos_sin, window_size, kv_cache, read_kv=None, kv_capture=None):
        B, T, C = x.size()

        # read_kv: per-layer committed (k, v) projections from cycle 1 of the
        #   internal verification loop. When given, attention is ANCHORED to
        #   the committed K/V (cycle >= 2 refines the stream, not the memory).
        # kv_capture: list that cycle 1 fills with its per-layer (k, v)
        #   projections (after VE gating + rotary), used as read_kv later.

        # Project the input to get queries, keys, and values
        # Shape: (B, T, H, D) - FA3's native layout, no transpose needed!
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Value residual (ResFormer): mix in value embedding with input-dependent gate per head
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))  # (B, T, n_kv_head), range (0, 3)
            v = v + gate.unsqueeze(-1) * ve

        # Apply Rotary Embeddings to queries and keys to get relative positional encoding
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k) # QK norm
        q = q * 1.2  # sharper attention (split scale between Q and K), TODO think through better
        k = k * 1.2

        # Cycle 1: optionally capture the committed K/V (after gating + rotary)
        if kv_capture is not None:
            kv_capture.append((k, v))

        # Flash Attention (FA3 or SDPA fallback)
        # window_size is (left, right) tuple: (N, 0) for causal, (-1, 0) for full context
        if kv_cache is None:
            # Training: causal attention with optional sliding window
            if read_kv is not None:
                k, v = read_kv[self.layer_idx]  # anchor to committed K/V
            y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        else:
            # Inference: use flash_attn_with_kvcache which handles cache management
            k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
            if read_kv is not None:
                # Refinement cycle: rewrite the committed K/V at the same cache
                # position (idempotent overwrite) instead of the fresh ones, so
                # attention stays anchored to the cycle-1 memory.
                k, v = read_kv[self.layer_idx]
            y = flash_attn.flash_attn_with_kvcache(
                q, k_cache, v_cache,
                k=k, v=v,
                cache_seqlens=kv_cache.cache_seqlens,
                causal=True,
                window_size=window_size,
            )
            # Advance position after last layer processes (cycle 1 only -
            # refinement cycles must not move the cache position)
            if self.layer_idx == kv_cache.n_layers - 1 and read_kv is None:
                kv_cache.advance(T)

        # Re-assemble the heads and project back to residual stream
        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        if config.n_layer <= 0 or config.moe_every <= 0:
            raise ValueError("n_layer and moe_every must be positive")
        self.attn = CausalSelfAttention(config, layer_idx)
        use_moe = (config.moe_num_experts > 0 and
                   ((layer_idx + 1) % config.moe_every == 0 or layer_idx == config.n_layer - 1))
        self.mlp = SparseMoE(
            config.n_embd, config.moe_num_experts, config.moe_expert_ratio,
            config.moe_top_k, config.moe_aux_loss_scale,
            config.moe_router_z_loss_scale,
        ) if use_moe else MLP(config)

    def forward(self, x, ve, cos_sin, window_size, kv_cache, read_kv=None, kv_capture=None):
        x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache, read_kv=read_kv, kv_capture=kv_capture)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        """
        NOTE a major footgun: this __init__ function runs in meta device context (!!)
        Therefore, any calculations inside here are shapes and dtypes only, no actual data.
        => We actually initialize all data (parameters, buffers, etc.) in init_weights() instead.
        """
        super().__init__()
        self.config = config
        self._last_hidden = None  # filled when expose_hidden=True (bridge use)
        # Compute per-layer window sizes for sliding window attention
        # window_size is (left, right) tuple: (-1, 0) for full context, (N, 0) for sliding window
        self.window_sizes = self._compute_window_sizes(config)
        # Hierarchical sliding window (meta-device safe: params init in init_weights)
        if getattr(config, "modality_window", False):
            from FoxSay.modality_window import SuperWindow
            self.super_window = SuperWindow.build(
                config.n_embd,
                sub_factors=getattr(config, "modality_sub_factors", None),
                smooth=getattr(config, "window_smooth", 0.9),
                init_params=False,
            )
        else:
            self.super_window = None
        # Sub-GAN cluster (meta-device safe: params init in init_weights)
        if getattr(config, "subgan_layers", None) is not None:
            from FoxSay.gan.subgans import SubGANCluster
            self.subgan_cluster = SubGANCluster(
                config.n_embd, layers=list(config.subgan_layers), init_params=False
            )
        else:
            self.subgan_cluster = None
        self._subgan_pass = 0
        # Mind (mathematical substrate), opt-in
        if getattr(config, "mind", False):
            from FoxSay.mind import Mind
            self.mind = Mind(
                config.n_embd,
                n_cycles=getattr(config, "loop_max_cycles", 1) or 1,
                subgan_ids=(self.subgan_cluster.subgans.keys()
                            if self.subgan_cluster is not None else ()),
            )
        else:
            self.mind = None
        # Pad vocab for efficiency (DDP, tensor cores). This is just an optimization - outputs are cropped in forward().
        # https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.resize_token_embeddings
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency")
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)]),
        })
        self.lm_head = Linear(config.n_embd, padded_vocab_size, bias=True)
        # bias=True: the logit baseline. The softmax has a translation degree
        # of freedom; a learnable bias absorbs the token-frequency prior (the
        # standard "necessary bias" of the output layer).
        # Per-layer learnable scalars (inspired by modded-nanogpt)
        # resid_lambdas: scales the residual stream at each layer (init 1.0 = neutral)
        # x0_lambdas: blends initial embedding back in at each layer (init 0.0 = disabled)
        # Separate parameters so they can have different optimizer treatment
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))   # fake init, real init in init_weights()
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))     # fake init, real init in init_weights()
        # Smear: mix previous token's embedding into current token (cheap bigram-like info)
        self.smear_gate = Linear(24, 1, bias=False)
        self.smear_lambda = nn.Parameter(torch.zeros(1))
        # Backout: subtract cached mid-layer residual before final norm to remove low-level features
        self.backout_lambda = nn.Parameter(0.2 * torch.ones(1))
        # Value embeddings (ResFormer-style): alternating layers, last layer always included
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({str(i): nn.Embedding(padded_vocab_size, kv_dim) for i in range(config.n_layer) if has_ve(i, config.n_layer)})
        # To support meta device initialization, we init the rotary embeddings here, but it's just "fake" meta tensors only.
        # As for rotary_seq_len, these rotary embeddings are pretty small/cheap in memory,
        # so let's just over-compute them by 10X, but assert fail if we ever reach that amount.
        # In the future we can dynamically grow the cache, for now it's fine.
        self.rotary_seq_len = config.sequence_len * 10 # 10X over-compute should be enough, TODO make nicer?
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False) # persistent=False means it's not saved to the checkpoint
        self.register_buffer("sin", sin, persistent=False)

        # Internal verification loop gate (ACT halting + adapter + mixture).
        # Params are initialized to exact neutral values in init_weights(), so
        # an untrained loop is bit-identical to the legacy model, and old
        # checkpoints load with neutral values via _patch_missing_keys.
        self.loop_gate = LoopGate(config.n_embd, max(1, config.loop_max_cycles)) if config.loop_max_cycles > 1 else None

    @torch.no_grad()
    def init_weights(self):
        """
        Initialize the full model in this one function for maximum clarity.

        wte (embedding):     normal, std=1.0
        lm_head:             normal, std=0.001
        for each block:
            attn.c_q:        uniform, std=1/sqrt(n_embd)
            attn.c_k:        uniform, std=1/sqrt(n_embd)
            attn.c_v:        uniform, std=1/sqrt(n_embd)
            attn.c_proj:     zeros
            mlp.c_fc:        uniform, std=1/sqrt(n_embd)
            mlp.c_proj:      zeros
        """

        # Embedding and unembedding
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=0.8)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        # bias branch: every bias in the model (lm_head, ve_gate, routers,
        # GAN heads) starts at zero -- baselines enter the game neutrally
        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                torch.nn.init.zeros_(m.bias)

        # Transformer blocks: uniform init with bound = sqrt(3) * std (same standard deviation as normal)
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5 # sqrt(3) multiplier makes sure Uniform achieves the same std as Normal
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s) # weights use Uniform to avoid outliers
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight) # projections are zero
            if isinstance(block.mlp, SparseMoE):
                block.mlp.init_weights()
            else:
                torch.nn.init.uniform_(block.mlp.c_fc.weight, -s * 0.4, s * 0.4)
                torch.nn.init.zeros_(block.mlp.c_proj.weight)
        # SuperWindow router: zero-init -> uniform routing -> static window at start
        if self.super_window is not None:
            torch.nn.init.zeros_(self.super_window.router.weight)
            torch.nn.init.zeros_(self.super_window.router.bias)
        # Sub-GANs: zero gate/refine-out -> pass-through at birth
        if self.subgan_cluster is not None:
            self.subgan_cluster.reset_parameters()
        # Mind: real-tensor init for its submodules
        if self.mind is not None:
            self.mind.init_weights()

        # Per-layer scalars
        # Per-layer resid init: stronger residual at early layers, weaker at deep layers
        n_layer = self.config.n_layer
        for i in range(n_layer):
            self.resid_lambdas.data[i] = 1.15 - (0.10 * i / max(n_layer - 1, 1))
        # Decaying x0 init: earlier layers get more input embedding blending
        for i in range(n_layer):
            self.x0_lambdas.data[i] = 0.20 - (0.15 * i / max(n_layer - 1, 1))

        # Smear/backout scalars and smear gate must be explicitly initialized 
        torch.nn.init.zeros_(self.smear_lambda)
        torch.nn.init.constant_(self.backout_lambda, 0.2)
        torch.nn.init.uniform_(self.smear_gate.weight, 0.0, 0.02)

        # Value embeddings (init like c_v: uniform with same std)
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)

        # Gate weights init with small positive values so gates start slightly above neutral
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.uniform_(block.attn.ve_gate.weight, 0.0, 0.02)

        # Rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin

        # Cast embeddings to COMPUTE_DTYPE: optimizer can tolerate reduced-precision
        # embeddings and it saves memory. Exception: fp16 requires fp32 embeddings
        # because GradScaler cannot unscale fp16 gradients.
        if COMPUTE_DTYPE != torch.float16:
            self.transformer.wte.to(dtype=COMPUTE_DTYPE)
            for ve in self.value_embeds.values():
                ve.to(dtype=COMPUTE_DTYPE)

        # Internal verification loop: neutral (zero-init) gate => the untrained
        # loop is bit-identical to the legacy model (cycle 1 halts with p~0.95,
        # adapter and mix_lambda are exactly zero).
        if self.loop_gate is not None:
            self.loop_gate.init_weights()

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=100000, device=None):
        # TODO: bump base theta more? e.g. 100K is more common more recently
        # autodetect the device from model embeddings
        if device is None:
            device = self.transformer.wte.weight.device
        # stride the channels
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        # stride the time steps
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        # calculate the rotation frequencies at each (time, channel) pair
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.to(COMPUTE_DTYPE), sin.to(COMPUTE_DTYPE)
        cos, sin = cos[None, :, None, :], sin[None, :, None, :] # add batch and head dims for later broadcasting
        return cos, sin

    def _compute_window_sizes(self, config):
        """
        Compute per-layer window sizes for sliding window attention.

        Returns list of (left, right) tuples for FA3's window_size parameter:
        - left: how many tokens before current position to attend to (-1 = unlimited)
        - right: how many tokens after current position to attend to (0 for causal)

        Pattern string is tiled across layers. Final layer always gets L (full context).
        Characters: L=long (full context), S=short (quarter context)
        """
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern), f"Invalid window_pattern: {pattern}. Use only S and L."
        # Map characters to window sizes
        long_window = config.sequence_len
        short_window = -(-long_window // 4 // 128) * 128  # ceil to FA3 tile size (2048 -> 768)
        char_to_window = {
            "L": (long_window, 0),
            "S": (short_window, 0),
        }
        # Tile pattern across layers
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        # Final layer always gets full context
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def get_device(self):
        return self.transformer.wte.weight.device

    def estimate_flops(self):
        """
        Return the estimated FLOPs per token for the model (forward + backward).
        Each matmul weight parameter contributes 2 FLOPs (multiply *, accumulate +) in forward, and 2X that in backward => 2+4=6.
        Cleanest explanation of this: https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4
        On top of that, 12 * h * q * effective_seq_len accounts for key @ query matmul flops inside attention.
        With sliding windows, effective_seq_len varies per layer (capped by window size).
        Ref: https://arxiv.org/abs/2204.02311 (PaLM paper).
        This is ~1% off from the exact formulas of Chinchilla paper, the difference is:
        - Chinchilla counts the embedding layer as flops (? weird, it's just a lookup => we ignore)
        - Chinchilla counts exp/sum/divide in attention softmax as flops (a little sus and very tiny => we ignore)
        """
        h, q, t = self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        # Sum attention FLOPs per layer, accounting for sliding window
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]  # (left, right) tuple, we use left
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
        num_flops_per_token = 6 * self.num_matmul_params() + attn_flops
        return num_flops_per_token

    def num_matmul_params(self):
        """Active matmul parameters per token (Top-K experts, not all experts)."""
        matmul_params = sum(m.weight.numel() for m in self.modules() if isinstance(m, Linear))
        matmul_params += sum(m.parameter_report()["active"] for m in self.modules()
                             if isinstance(m, SparseMoE))
        return matmul_params

    def estimate_decode_flops(self, context_len):
        """
        Forward FLOPs to decode one token at a given context length during inference:
        2 FLOPs per matmul param, plus attention over min(context, window) per layer.
        """
        h = self.config.n_head
        q = self.config.n_embd // self.config.n_head
        attn_flops = sum(4 * h * q * min(context_len, window) for window, _ in self.window_sizes)
        decode_flops = 2 * self.num_matmul_params() + attn_flops
        return decode_flops

    def estimate_prefill_flops(self, num_tokens):
        """Forward FLOPs to prefill a prompt: causal, so token t attends to min(t, window)."""
        h = self.config.n_head
        q = self.config.n_embd // self.config.n_head
        attn_flops = 0
        for window, _ in self.window_sizes:
            w = min(window, num_tokens)
            attended_tokens = w * (w + 1) // 2 + (num_tokens - w) * w # ramp up to w, then flat
            attn_flops += 4 * h * q * attended_tokens
        prefill_flops = 2 * self.num_matmul_params() * num_tokens + attn_flops
        return prefill_flops

    def kv_bytes_per_token(self):
        """Bytes to *store* one token of KV cache during inference, per row (all layers)."""
        head_dim = self.config.n_embd // self.config.n_head
        kv_dtype_bytes = COMPUTE_DTYPE.itemsize # the KV cache is kept in the compute dtype
        return self.config.n_layer * 2 * self.config.n_kv_head * head_dim * kv_dtype_bytes

    def kv_read_bytes(self, context_len):
        """Bytes of KV cache *read* by one decode step at a given context length, per row.
        Sliding window layers only attend to (and read) the last `window` tokens."""
        head_dim = self.config.n_embd // self.config.n_head
        kv_dtype_bytes = COMPUTE_DTYPE.itemsize
        total = 0
        for window, _ in self.window_sizes:
            total += 2 * self.config.n_kv_head * head_dim * kv_dtype_bytes * min(context_len, window)
        return total

    def num_scaling_params(self):
        """
        Return detailed parameter counts for scaling law analysis.
        Different papers use different conventions:
        - Kaplan et al. excluded embedding parameters
        - Chinchilla included all parameters
        Ref: https://arxiv.org/abs/2203.15556 (Chinchilla paper)
        Ref: https://arxiv.org/abs/2001.08361 (Kaplan et al. original scaling laws paper)

        Returns a dict with counts for each parameter group, so downstream analysis
        can experiment with which combination gives the cleanest scaling laws.
        """
        # Count each group separately (mirrors the grouping in setup_optimizers)
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        inactive_experts = sum(
            m.parameter_report()["total"] - m.parameter_report()["active"]
            for m in self.modules() if isinstance(m, SparseMoE)
        )
        transformer_active = transformer_matrices - inactive_experts
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel() + self.smear_gate.weight.numel() + self.smear_lambda.numel() + self.backout_lambda.numel()
        total = wte + value_embeds + lm_head + transformer_matrices + scalars
        assert total == sum(p.numel() for p in self.parameters()), "Parameter count mismatch"
        return {
            'wte': wte,
            'value_embeds': value_embeds,
            'lm_head': lm_head,
            'transformer_matrices': transformer_matrices,
            'transformer_active': transformer_active,
            'scalars': scalars,
            'total': total,
            'active_total': total - inactive_experts,
        }

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0, scalar_lr=0.5):
        model_dim = self.config.n_embd

        # Separate out all parameters into groups
        matrix_params = list(self.transformer.h.parameters())
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = [p for p in self.lm_head.parameters() if p.dim() >= 2]
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        smear_params = [self.smear_gate.weight, self.smear_lambda, self.backout_lambda]
        loop_params = list(self.loop_gate.parameters()) if self.loop_gate is not None else []
        lm_head_bias_params = [p for p in self.lm_head.parameters() if p.dim() < 2]
        assert len(list(self.parameters())) == len(matrix_params) + len(embedding_params) + len(lm_head_params) + len(value_embeds_params) + len(resid_params) + len(x0_params) + len(smear_params) + len(loop_params) + len(lm_head_bias_params)

        # Scale the LR for the AdamW parameters by ∝1/√dmodel (tuned for 768 dim model)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")

        # Build param_groups with all required fields explicit
        param_groups = [
            # AdamW groups (embeddings, lm_head, scalars)
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=(0.8, 0.96), eps=1e-10, weight_decay=0.01),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.001),
            dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale * 0.5, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.01),
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.05),
            dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),  # higher beta1 for x0
            dict(kind='adamw', params=smear_params, lr=0.2, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0),
        ]
        # Internal verification loop gate params (zero-init: gradients wake them
        # up only when the loop actually runs). Adapter/readout are matrices ->
        # Muon; biases + mix_lambda are scalars -> AdamW.
        if loop_params:
            loop_matrix = [p for p in loop_params if p.dim() >= 2]
            loop_scalar = [p for p in loop_params if p.dim() < 2]
            if loop_matrix:
                for shape in sorted({p.shape for p in loop_matrix}):
                    group_params = [p for p in loop_matrix if p.shape == shape]
                    param_groups.append(dict(
                        kind='muon', params=group_params, lr=matrix_lr * 0.5,
                        momentum=0.95, ns_steps=5, beta2=0.9, weight_decay=weight_decay,
                    ))
            if loop_scalar:
                param_groups.append(dict(kind='adamw', params=loop_scalar, lr=scalar_lr * 0.01, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0))
        # Muon groups (2D matrix params only, grouped by shape for stacking).
        # Forward-looking bias routing: any 1D param (ve_gate bias, future
        # layer biases) automatically lands in AdamW -- Muon never sees a
        # non-matrix, so adding bias anywhere stays safe.
        muon_matrix = [p for p in matrix_params if p.dim() >= 2]
        bias_params = [p for p in matrix_params if p.dim() < 2] + \
                      [p for p in self.lm_head.parameters() if p.dim() < 2]
        if bias_params:
            param_groups.append(dict(
                kind='adamw', params=bias_params, lr=scalar_lr * 0.01,
                betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0,
            ))
        for shape in sorted({p.shape for p in muon_matrix}):
            group_params = [p for p in muon_matrix if p.shape == shape]
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.9, weight_decay=weight_decay,
            ))

        optimizer = MuonAdamW(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
        B, T = idx.size()
        if self.subgan_cluster is not None:
            self._subgan_pass = 0  # commit pass = cycle 0; each run_trunk = +1

        # Grab the rotary embeddings for the current sequence length (they are of shape (1, seq_len, 1, head_dim/2))
        assert T <= self.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
        assert idx.device == self.cos.device, f"Rotary embeddings and idx are on different devices: {idx.device} != {self.cos.device}"
        assert self.cos.dtype == COMPUTE_DTYPE, f"Rotary embeddings must be in {COMPUTE_DTYPE}, got {self.cos.dtype}"
        # if kv cache exists, we need to offset the rotary embeddings to the current position in the cache
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T] # truncate cache to current sequence length

        # Embed the tokens
        x = self.transformer.wte(idx) # embed current token
        x = x.to(COMPUTE_DTYPE) # ensure activations are in compute dtype (no-op usually, but active for fp16 code path)
        x = norm(x)

        # Smear: mix previous token's embedding into current position (cheap bigram info)
        if kv_cache is None:
            # Training / naive generate: full sequence available, use fast slice
            assert T > 1, "Training forward pass should have T > 1"
            gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
            x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
        else:
            # KV cache inference: read prev embedding from cache, store current for next step
            x_pre_smear = kv_cache.prev_embedding
            kv_cache.prev_embedding = x[:, -1:, :]
            if T > 1:
                # Prefill: apply smear to positions 1+, same as training
                gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
                x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
            elif x_pre_smear is not None:
                # Decode: single token, use cached prev embedding
                gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, :, :24]))
                x = x + gate * x_pre_smear

        # Forward the trunk of the Transformer
        x0 = x  # save initial normalized embedding for x0 residual
        n_layer = self.config.n_layer
        backout_layer = n_layer // 2  # cache at halfway point
        moe_aux_losses = []

        def run_trunk(x_in, read_kv=None, kv_capture=None):
            """One pass over the transformer trunk. Returns (stream, backout_state)."""
            x = x_in
            x_backout = None
            if self.subgan_cluster is not None:
                self._subgan_pass += 1
                self.subgan_cluster.begin_cycle(self._subgan_pass - 1)
            for i, block in enumerate(self.transformer.h):
                x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
                ve = self.value_embeds[str(i)](idx).to(x.dtype) if str(i) in self.value_embeds else None
                window = self.window_sizes[i]
                if self.super_window is not None:
                    window = self.super_window(x, window)  # modality-specialized unified window
                x = block(x, ve, cos_sin, window, kv_cache, read_kv=read_kv, kv_capture=kv_capture)
                aux = getattr(block.mlp, "last_aux_loss", None)
                if aux is not None:
                    moe_aux_losses.append(aux)
                if self.subgan_cluster is not None:
                    x = self.subgan_cluster.maybe_think(x, i, n_layer)
                if i == backout_layer:
                    x_backout = x
            return x, x_backout

        def post_backout(x_stream, x_backout):
            # Subtract mid-layer residual to remove low-level features before logit projection
            if x_backout is not None:
                x_stream = x_stream - self.backout_lambda.to(x_stream.dtype) * x_backout
            return x_stream

        # ------------------------- internal verification loop -------------------------
        # Cycle 1 is the legacy commit pass: it writes K/V (training: in-tensor,
        # decode: into the KV cache) exactly like the original model. Cycles >= 2
        # refine the latent stream while attention stays anchored to the
        # committed K/V (read_kv). Zero-initialized gate/mix parameters make the
        # untrained loop bit-identical to the legacy forward pass.
        loop_gate = self.loop_gate
        if loop_gate is None:
            loop_cycles = 1
        else:
            forced = getattr(self, "_loop_depth", None)
            if forced is not None:
                loop_cycles = max(1, min(int(forced), loop_gate.max_cycles))
            else:
                # Training: run every cycle, soft ACT mixture over states.
                # Inference: max_cycles is the BUDGET; per-row hard halting
                # (p_stop >= threshold) stops early - at neutral init that is
                # cycle 1 for every row, i.e. the legacy behavior.
                loop_cycles = loop_gate.max_cycles

        if loop_cycles <= 1:
            x, x_backout = run_trunk(x)
            x = post_backout(x, x_backout)
            if loop_gate is not None:
                self._last_loop_cycles = None
        else:
            committed_kv = []
            x_c1, x_backout_c1 = run_trunk(x, kv_capture=committed_kv)  # commit pass
            s_commit = post_backout(x_c1, x_backout_c1)

            if self.training:
                # Soft ACT relaxation: run all cycles, mixture over cycle states.
                states = [s_commit]
                probs = [loop_gate.stop_probs(x_c1, 0)]
                x_prev = x_c1
                for c in range(1, loop_cycles):
                    x_prev = loop_gate.adapt(x_prev)  # residual adapter: x + adapter(x)
                    x_c, x_bo_c = run_trunk(x_prev, read_kv=committed_kv)
                    if self.mind is not None:
                        self.mind.observe(x_c, cycle=c,
                                          depth_fraction=(c + 1) / loop_cycles)
                    states.append(post_backout(x_c, x_bo_c))
                    probs.append(loop_gate.stop_probs(x_c, c))
                    x_prev = x_c
                weights = loop_gate.act_weights(probs)  # ACT halting distribution
                mix = None
                for w, st in zip(weights, states):
                    term = w.view(-1, 1, 1).to(st.dtype) * st
                    mix = term if mix is None else mix + term
                # Zero-init blend: out = s_commit + lambda * (mixture - s_commit)
                x = loop_gate.blend(s_commit, mix)
                self._last_loop_cycles = None
            else:
                # Hard per-row halting at inference (ACT test-time semantics):
                # rows keep refining until their p_stop crosses the threshold.
                # At neutral init p_stop(cycle 1) ~ 0.95 >= threshold => every
                # row halts after cycle 1 => bit-identical legacy behavior.
                threshold = float(self.config.loop_stop_threshold)
                if self.mind is not None:
                    threshold += self.mind.stop_threshold_shift()
                B_row = x_c1.size(0)
                p_c1 = loop_gate.stop_probs(x_c1, 0)
                halted = p_c1 >= threshold
                s = s_commit
                depth = torch.ones(B_row, dtype=torch.long, device=x_c1.device)
                x_prev = x_c1
                c = 1
                while c < loop_cycles and not bool(halted.all()):
                    x_prev = loop_gate.adapt(x_prev)
                    x_c, x_bo_c = run_trunk(x_prev, read_kv=committed_kv)
                    s_c = post_backout(x_c, x_bo_c)
                    p_c = loop_gate.stop_probs(x_c, c)
                    s_refined = loop_gate.blend(s_commit, s_c)
                    keep = halted.view(-1, 1, 1)
                    s = torch.where(keep, s, s_refined.to(s.dtype))
                    depth = torch.where(halted, depth, torch.full_like(depth, c + 1))
                    halted = halted | (p_c >= threshold)
                    x_prev = x_c
                    c += 1
                x = s
                self._last_loop_cycles = depth  # per-row halting cycle for the caller (engine/RL)
        # ---------------------------------------------------------------------------------

        x = norm(x)

        # ---- dual-brain bridge hook: expose the final hidden state ---------
        # Zero-cost unless requested: the corpus callosum (engine side) reads
        # this reference to feed the right brain. Off by default (legacy).
        if getattr(self, "expose_hidden", False):
            self._last_hidden = x

        # Logits and log-softmax use at least fp32; fp64 is opt-in for final-
        # convergence work. Chunking bounds the dominant B*T*V activation.
        try:
            loss_dtype = {"float32": torch.float32, "float64": torch.float64}[self.config.loss_dtype]
        except KeyError as e:
            raise ValueError("loss_dtype must be 'float32' or 'float64'") from e

        def project_logits(hidden):
            projected = self.lm_head(hidden)[..., :self.config.vocab_size].to(loss_dtype)
            cap = float(self.config.logit_softcap)
            return cap * torch.tanh(projected / cap) if cap > 0 else projected

        if targets is None:
            return project_logits(x)

        flat_targets = targets.reshape(-1)
        chunk = int(self.config.loss_chunk_size)
        if chunk > 0 and loss_reduction in ("mean", "sum"):
            flat_x = x.reshape(-1, x.size(-1))
            parts = []
            for start in range(0, flat_x.size(0), chunk):
                part_logits = project_logits(flat_x[start:start + chunk])
                parts.append(F.cross_entropy(
                    part_logits, flat_targets[start:start + chunk],
                    ignore_index=-1, reduction="sum",
                ))
            loss = torch.stack(parts).sum()
            if loss_reduction == "mean":
                loss = loss / flat_targets.ne(-1).sum().clamp_min(1)
        else:
            reduction = "sum" if loss_reduction == "mean" else loss_reduction
            loss = F.cross_entropy(
                project_logits(x).reshape(-1, self.config.vocab_size), flat_targets,
                ignore_index=-1, reduction=reduction,
            )
            if loss_reduction == "mean":
                loss = loss / flat_targets.ne(-1).sum().clamp_min(1)
        if moe_aux_losses and loss_reduction in ("mean", "sum"):
            has_targets = flat_targets.ne(-1).any().to(loss.dtype)
            loss = loss + has_targets * torch.stack(moe_aux_losses).sum().to(loss.dtype)
        return loss

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        """
        Naive autoregressive streaming inference.
        To make it super simple, let's assume:
        - batch size is 1
        - ids and the yielded tokens are simple Python lists and ints
        """
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device) # add batch dim
        for _ in range(max_tokens):
            logits = self.forward(ids) # (B, T, vocab_size)
            logits = logits[:, -1, :] # (B, vocab_size)
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, next_ids), dim=1)
            token = next_ids.item()
            yield token
