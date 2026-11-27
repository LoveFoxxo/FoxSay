"""
FSDP (ZeRO-3 style) *weight* sharding for FoxSay, layered on top of FoxSay's
ZeRO-2 optimizer-state sharding (FoxSay/optim.py).

Why: MuonAdamW shards optimizer state and gradients, but the parameters themselves
are replicated on every rank (fp32 matrices + bf16 embeddings). That caps the
trainable model size per node. This module shards the weights too, so per-rank
persistent memory becomes O(model_size / world_size):

What is sharded, and how
------------------------
1. Muon params (block matrices: c_q/c_k/c_v/c_proj/c_fc/c_proj/ve_gate):
   Each param is owned *whole* by exactly one rank (param-index chunking, using
   the exact same grouping math as the optimizer's reduce-scatter, see
   fsdp/shard_math.py). The owner keeps the full master in `master_dtype`
   (fp32, or fp64 for the "precision to 64" mod). Forward: the owner broadcasts
   a compute-dtype copy to every rank; the copies are dropped once the owning
   block's backward finished. Optimizer step: only the owner runs Muon on its
   params ( grads are reduce-scattered exactly as in stock FoxSay) and writes
   the update straight into its master. The stock optimizer's final
   all-gather-back of updated params is *removed* - weights stay sharded.

2. AdamW params (wte, value_embeds, lm_head - the memory hogs):
   Each AdamW param *group* is flattened into one buffer, zero-padded to a
   multiple of world_size, and chunked evenly. Each rank permanently holds only
   its chunk of the master (fp32/fp64) plus its chunk of exp_avg/exp_avg_sq.
   Forward: one all-gather rebuilds the flat compute-dtype buffer; each param
   becomes a view into it. Backward: full-size grads accumulate as usual; the
   optimizer copies them into the flat buffer, reduce-scatters (AVG), and runs
   the same fused AdamW kernel on the local chunk. No all-gather, ever.
   Flat chunking (unlike stock dim-0 slicing) works for ANY world size, e.g.
   32768 rows across 100 GPUs.

3. Tiny AdamW groups (< flat_min_numel elements: resid_lambdas, x0_lambdas,
   smear scalars) stay replicated and keep the stock all-reduce path.

Mechanics
---------
Model structure is NOT modified: no wrapper modules, state_dict keys unchanged.
Each "unit" module (wte, value_embeds[i], block[i], lm_head) gets a
forward pre-hook (dynamo-disabled so torch.compile graph-breaks around it) that
materializes its params as views into freshly-gathered buffers, and each param
gets a post-accumulate-grad hook that restores empty storage once the whole
unit's backward has completed. Grad tensors survive (the optimizer needs them
full-size; they are freed by model.zero_grad as usual).

Checkpoint compatibility
------------------------
- `FullySharded.consolidated_state_dict()` rebuilds the standard full model
  state dict (original dtypes restored), so rank0 keeps writing plain
  `model_<step>.pt` checkpoints loadable by any FoxSay script.
- Optimizer state is saved per-rank (same convention as stock FoxSay), but in
  the FSDP layout (flat chunks). Resuming therefore requires the same world
  size. Loading a *stock* checkpoint's weights works fine (weights load
  pre-shard, then masters are carved out of them).

Usage (scripts/base_train.py):
    torchrun --nproc_per_node=8 -m scripts.base_train --fsdp
    torchrun --nproc_per_node=64 -m scripts.base_train --fsdp --fsdp-master-dtype float64

Limitations / notes
-------------------
- fp16 (GradScaler) and fp8 are not supported under FSDP.
- Gradients are still full-size during the backward of each step (2 bytes/param
  in bf16 compute). Sharding grads at unit granularity is a possible follow-up.
- One all-gather (AdamW units) or k broadcasts (Muon units) per unit per
  micro-batch; one-unit-ahead prefetch overlaps comm with compute.
- torch.compile: hooks are dynamo-disabled -> one graph break per unit per
  forward. Correct, slightly less fused.
- Optimizer checkpoints are world-size-specific (same as stock FoxSay's
  rank-sharded optimizer state, enforced here with a clear error).
"""

import contextlib
import functools

import torch
import torch.distributed as dist

from FoxSay.common import COMPUTE_DTYPE, is_ddp_initialized, print0
from FoxSay.fsdp.shard_math import (
    ceil_div,
    flat_chunk_bounds,
    flat_rank_slice,
    muon_chunk_size,
    muon_owned_range,
)
from FoxSay.optim import MuonAdamW, adamw_step_fused, muon_step_fused, dim0_shard_ok

try:
    # torch >= 2.4 spelling; falls back on older versions
    _compiler_disable = torch.compiler.disable
except AttributeError:  # pragma: no cover
    from torch._dynamo import disable as _compiler_disable


# -----------------------------------------------------------------------------
# registry records


class ParamShard:
    """Book-keeping for one parameter under FSDP."""

    __slots__ = (
        "kind", "param", "shape", "numel", "orig_dtype",
        "owner",           # muon only: rank owning the full master
        "master",          # muon only: full master tensor (owner rank only)
        "group",           # adamw_flat only: the owning AdamWFlatGroup
        "offset",          # adamw_flat only: offset into the group's flat buffer
        "cur_buf",         # transient compute-dtype storage during forward/backward
        "cur_gather",
    )

    def __init__(self, kind, param, shape, numel, orig_dtype):
        self.kind = kind          # 'muon' | 'adamw_flat' | 'adamw_small'
        self.param = param
        self.shape = tuple(shape)
        self.numel = numel
        self.orig_dtype = orig_dtype
        self.owner = None
        self.master = None
        self.group = None
        self.offset = None
        self.cur_buf = None
        self.cur_gather = None


class AdamWFlatGroup:
    """One AdamW param group flattened into a single padded, chunked buffer."""

    def __init__(self, params, hypergroup, world_size, master_dtype, compute_dtype, reduce_dtype, device):
        self.params = list(params)
        self.hypergroup = hypergroup  # the optimizer param_groups dict (lr, betas, ...)
        self.master_dtype = master_dtype
        self.compute_dtype = compute_dtype
        self.reduce_dtype = reduce_dtype  # dtype the gradient sum runs in (>= fp32, = fp64 for fp64 masters)
        self.device = device
        self.offset_of = {}
        offset = 0
        for p in self.params:
            self.offset_of[p] = (offset, p.numel())
            offset += p.numel()
        self.total_numel = offset
        self.chunk_len, self.padded_numel = flat_chunk_bounds(self.total_numel, world_size)
        # persistent per-rank state (filled in by FullySharded.shard)
        self.master_chunk = None      # (chunk_len,) master dtype - the authoritative storage
        self.chunk_compute = None     # (chunk_len,) compute dtype - all-gather source, cached
        self.grad_flat = None         # (padded_numel,) reduce dtype - reduce-scatter input
        self.grad_chunk = None        # (chunk_len,) reduce dtype - reduce-scatter output (exact mean)
        self.exp_avg = None           # (chunk_len,) master dtype
        self.exp_avg_sq = None        # (chunk_len,) master dtype
        self.step = 0


class Unit:
    """A sharding unit = one module whose params are gathered/free'd together."""

    __slots__ = ("name", "module", "shards", "order", "launched", "handles", "grad_count")

    def __init__(self, name, module, shards, order):
        self.name = name
        self.module = module
        self.shards = shards
        self.order = order
        self.launched = False
        self.handles = []
        self.grad_count = 0


class _ScaledMean:
    """Async-work wrapper: on wait(), finishes the reduce-scatter SUM and applies
    the single exact division that turns the sum into the mean (see
    FullySharded._reduce_scatter_avg). Callers keep the plain `future.wait()`."""

    __slots__ = ("future", "out", "n")

    def __init__(self, future, out, n):
        self.future = future
        self.out = out
        self.n = n

    def wait(self):
        self.future.wait()
        self.out.div_(self.n)  # one identical rounding on every rank


# -----------------------------------------------------------------------------
# the sharding manager


class FullySharded:
    """
    Shards a GPT's weights across ranks and swaps in an FSDP-aware optimizer.

    Usage:
        model = ...  # built + init_weights() (+ resumed) as usual
        optimizer = model.setup_optimizer(...)   # stock MuonAdamW
        fsdp = FullySharded(model, optimizer, master_dtype=torch.float64)
        fsdp.shard()
        optimizer = fsdp.optimizer                # FSDPMuonAdamW, same param_groups
        # ... training loop unchanged ...
        save_checkpoint(..., fsdp.consolidated_state_dict(), optimizer.state_dict(), ...)
    """

    FLAT_GROUP_MIN_NUMEL = 4096  # groups smaller than this stay replicated

    def __init__(self, model, optimizer, master_dtype=torch.float32, flat_min_numel=None):
        assert isinstance(optimizer, MuonAdamW), "FullySharded expects the stock MuonAdamW optimizer"
        assert master_dtype in (torch.float32, torch.float64), "master_dtype must be fp32 or fp64"
        self.model = model
        self._orig_optimizer = optimizer
        self.master_dtype = master_dtype
        # Gradient-sum dtype. >= fp32 always: bf16 grad buffers lose low bits on
        # every accumulate/average round-trip ("more GPUs => worse precision" root
        # cause #2). = fp64 under fp64 masters so the reduced mean is exact.
        self.reduce_dtype = torch.float64 if master_dtype == torch.float64 else torch.float32
        if flat_min_numel is not None:
            self.FLAT_GROUP_MIN_NUMEL = flat_min_numel
        if is_ddp_initialized():
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
        else:
            self.world_size = 1
            self.rank = 0
        self.device = next(model.parameters()).device
        self._comm_stream = torch.cuda.Stream(device=self.device) if self.device.type == "cuda" else None
        self._gloo_no_reduce_scatter = False  # set True if backend lacks reduce_scatter_tensor
        self._gloo_no_all_gather_base = False
        # filled by shard():
        self.registry = {}            # param -> ParamShard
        self.adamw_flat_groups = []
        self.units = []
        self._unit_by_module = {}
        self._param_keys = {}         # param -> state_dict key name (captured pre-shard)
        self._shards_in_order = []
        self.optimizer = None         # FSDPMuonAdamW, set by shard()
        self._sharded = False

    # ------------------------------------------------------------------ shard

    @torch.no_grad()
    def shard(self):
        assert not self._sharded, "FullySharded.shard() called twice"
        self._classify()
        self._extract_masters()
        self._build_units()
        self._register_hooks()
        self.optimizer = FSDPMuonAdamW(self._orig_optimizer.param_groups, self)
        self._sharded = True
        self._print_stats()

    def _classify(self):
        registry = self.registry
        for group in self._orig_optimizer.param_groups:
            kind = group["kind"]
            params = group["params"]
            if kind == "muon":
                chunk_size = muon_chunk_size(len(params), self.world_size)
                for idx, p in enumerate(params):
                    shard = ParamShard("muon", p, p.shape, p.numel(), p.dtype)
                    shard.owner = idx // chunk_size  # same math as the optimizer's reduce-scatter
                    registry[p] = shard
            elif kind == "adamw":
                total = sum(p.numel() for p in params)
                if total < self.FLAT_GROUP_MIN_NUMEL:
                    for p in params:
                        registry[p] = ParamShard("adamw_small", p, p.shape, p.numel(), p.dtype)
                else:
                    fg = AdamWFlatGroup(params, group, self.world_size, self.master_dtype, COMPUTE_DTYPE, self.reduce_dtype, self.device)
                    self.adamw_flat_groups.append(fg)
                    for p in params:
                        shard = ParamShard("adamw_flat", p, p.shape, p.numel(), p.dtype)
                        shard.group = fg
                        shard.offset, shard.numel = fg.offset_of[p]
                        registry[p] = shard
            else:
                raise ValueError(f"Unknown optimizer kind: {kind}")
        self._shards_in_order = list(registry.values())
        self._param_keys = {p: name for name, p in self.model.named_parameters()}

    @torch.no_grad()
    def _extract_masters(self):
        """Carve the (identically initialized on every rank) local params into
        per-rank master storage, then empty the param tensors."""
        empty = functools.partial(torch.empty, 0, device=self.device)
        for shard in self._shards_in_order:
            p = shard.param
            if shard.kind == "muon":
                if self.world_size == 1 or shard.owner == self.rank:
                    # .to() returns the same tensor when dtypes match => zero-copy fp32 path
                    shard.master = p.data.detach().to(self.master_dtype)
                    p.data = shard.master  # keep the master as the param's home storage
                else:
                    shard.master = None  # non-owners must NOT keep a full master copy
                    p.data = empty(dtype=shard.orig_dtype)
            elif shard.kind == "adamw_flat":
                g = shard.group
                if g.master_chunk is None:
                    flat_full = torch.cat([p.data.detach().reshape(-1).to(g.master_dtype) for p in g.params])
                    lo, hi = flat_rank_slice(g.total_numel, self.world_size, self.rank)
                    # this rank's slice of the *padded* flat buffer is [lo, hi); master_chunk
                    # stores it at local offset 0, so slot i == global slot lo + i.
                    g.master_chunk = torch.zeros(g.chunk_len, dtype=g.master_dtype, device=self.device)
                    n_local = max(0, min(hi, g.total_numel) - lo)  # 0 on padding-only ranks (odd world sizes)
                    g.master_chunk[:n_local] = flat_full[lo : lo + n_local]
                    g.chunk_compute = torch.empty(g.chunk_len, dtype=g.compute_dtype, device=self.device)
                    g.grad_flat = torch.zeros(g.padded_numel, dtype=g.reduce_dtype, device=self.device)
                    g.grad_chunk = torch.empty(g.chunk_len, dtype=g.reduce_dtype, device=self.device)
                    g.exp_avg = torch.zeros(g.chunk_len, dtype=g.master_dtype, device=self.device)
                    g.exp_avg_sq = torch.zeros_like(g.exp_avg)
                    g.step = 0
                    del flat_full  # only this rank's chunk survives
                p.data = empty(dtype=shard.orig_dtype)
            else:  # adamw_small: keep everything as-is (replicated)
                pass

    def _build_units(self):
        model = self.model
        registry = self.registry

        def add_unit(name, module):
            shards = []
            for _, p in module.named_parameters(recurse=True):
                shard = registry.get(p)
                if shard is not None and shard.kind in ("muon", "adamw_flat"):
                    shards.append(shard)
            assert shards, f"unit {name} has no sharded params"
            unit = Unit(name, module, shards, order=len(self.units))
            self._unit_by_module[module] = unit
            self.units.append(unit)

        # RightBrain uses the same owner-sharding path, one unit per block.
        if not hasattr(model, "transformer"):
            blocks = getattr(model, "blocks", None)
            if blocks is None:
                raise TypeError("FullySharded expects GPT or RightBrain")
            for i, block in enumerate(blocks):
                add_unit(f"right.block[{i}]", block)
            return

        # Units in *forward-call order* (GPT.forward): wte -> per layer: ve, block -> lm_head.
        # The one-unit-ahead prefetch relies on this ordering.
        add_unit("wte", model.transformer.wte)
        for i, block in enumerate(model.transformer.h):
            ve_key = str(i)
            if ve_key in model.value_embeds:
                add_unit(f"value_embeds[{i}]", model.value_embeds[ve_key])
            add_unit(f"block[{i}]", block)
        add_unit("lm_head", model.lm_head)
        # Internal verification loop (see FoxSay/gan): the gate's readout and
        # adapter are called as CHILD modules (loop_gate.readout(...) / .adapt),
        # so each needs its OWN unit - a unit on loop_gate itself would never
        # fire. The muon-sharded matrices (readout.weight, adapter.weight) must
        # materialize via all-gather exactly like trunk params; the small
 # replicated params (biases, mix_lambda, cycle_biases) keep their data and
        # need no unit. Without these units, non-owner ranks hold empty weights
        # => rank-divergent crashes => all-gather deadlock (caught by stage 9).
        lg = getattr(model, "loop_gate", None)
        if lg is not None:
            add_unit("loop_gate.readout", lg.readout)
            add_unit("loop_gate.adapter", lg.adapter)

    def _register_hooks(self):
        pre_hook = _compiler_disable(self._pre_forward)
        for unit in self.units:
            unit.module.register_forward_pre_hook(pre_hook)
            for shard in unit.shards:
                hook = functools.partial(self._on_param_grad, unit)
                reg = getattr(shard.param, "register_post_accumulate_grad_hook", None)
                if reg is not None:
                    reg(hook)
                else:  # older torch: leaf hook fires once per backward, same semantics here
                    shard.param.register_hook(lambda grad, _p=shard.param, _u=unit: self._on_param_grad(_u, _p))

    # ------------------------------------------------------- forward/backward

    def _pre_forward(self, module, args):
        """Forward pre-hook: materialize this unit's params as views into fresh
        compute-dtype buffers, then prefetch the next unit in call order."""
        unit = self._unit_by_module[module]
        self._activate_unit(unit)
        next_order = unit.order + 1
        if next_order < len(self.units):
            self._launch_comms(self.units[next_order])
        return None

    def _on_param_grad(self, unit, param):
        """post-accumulate-grad hook: once the whole unit's backward is done,
        release the transient buffers and restore empty/master storage."""
        unit.grad_count += 1
        if unit.grad_count >= len(unit.shards):
            self._free_unit(unit)

    @torch.no_grad()
    def _launch_comms(self, unit):
        if unit.launched:
            return
        unit.launched = True
        unit.handles = []
        use_dist = self.world_size > 1
        stream_ctx = (
            torch.cuda.stream(self._comm_stream) if self._comm_stream is not None else contextlib.nullcontext()
        )
        if self._comm_stream is not None:
            # masters were last written by the optimizer on the compute stream
            self._comm_stream.wait_stream(torch.cuda.current_stream())
        with stream_ctx:
            for shard in unit.shards:
                if shard.kind == "adamw_flat":
                    g = shard.group
                    g.chunk_compute.copy_(g.master_chunk)  # cast this rank's chunk -> compute dtype
                    if use_dist:
                        out = torch.empty(g.padded_numel, dtype=g.compute_dtype, device=self.device)
                        handle = self._all_gather_flat(out, g.chunk_compute)
                        if handle is not None:
                            unit.handles.append(handle)
                        shard.cur_gather = out
                    else:
                        shard.cur_gather = g.chunk_compute  # ws == 1: chunk == full flat buffer
                elif shard.kind == "muon":
                    buf = torch.empty(shard.numel, dtype=COMPUTE_DTYPE, device=self.device)
                    if shard.owner == self.rank:
                        buf.copy_(shard.master.view(-1))  # cast master -> compute dtype
                    if use_dist:
                        handle = dist.broadcast(buf, src=shard.owner, async_op=True)
                        unit.handles.append(handle.get_future() if hasattr(handle, "get_future") else handle)
                    shard.cur_buf = buf

    def _wait_comms(self, unit):
        if self._comm_stream is not None:
            cur = torch.cuda.current_stream()
            cur.wait_stream(self._comm_stream)
            for shard in unit.shards:
                if shard.cur_gather is not None:
                    shard.cur_gather.record_stream(cur)
                if shard.cur_buf is not None:
                    shard.cur_buf.record_stream(cur)
        else:
            for handle in unit.handles:
                handle.wait()
        unit.handles = []

    @torch.no_grad()
    def _activate_unit(self, unit):
        if not unit.launched:
            self._launch_comms(unit)  # prefetch didn't happen (e.g. first call)
        self._wait_comms(unit)
        unit.launched = False
        for shard in unit.shards:
            if shard.kind == "adamw_flat":
                shard.param.data = shard.cur_gather[shard.offset : shard.offset + shard.numel].view(shard.shape)
            else:  # muon
                shard.param.data = shard.cur_buf.view(shard.shape)
        unit.grad_count = 0

    @torch.no_grad()
    def _free_unit(self, unit):
        # The latent bug this fixes: a unit can be speculatively relaunched by
        # the one-ahead prefetch (e.g. readout/adapter of the loop gate, which
        # sit AFTER lm_head in unit order but are re-entered every cycle) and
        # then freed by the backward grad hook. Without waiting the stray
        # handles and resetting `launched`, the next activation would skip
        # the launch and bind params to a nulled cur_buf/cur_gather.
        if unit.handles:
            self._wait_comms(unit)
        unit.launched = False
        empty = functools.partial(torch.empty, 0, device=self.device)
        for shard in unit.shards:
            if shard.kind == "muon" and (self.world_size == 1 or shard.owner == self.rank):
                shard.param.data = shard.master  # home storage again
            else:
                shard.param.data = empty(dtype=shard.orig_dtype)
            shard.cur_gather = None  # drop refs so the gathered storage can be reclaimed
            shard.cur_buf = None
        unit.grad_count = 0

    @torch.no_grad()
    def free_all(self):
        """Restore all params to their home storage and drop gathered buffers.
        Call after forward-only passes (eval / sampling), which never trigger
        the backward hooks."""
        for unit in self.units:
            self._free_unit(unit)

    # ------------------------------------------------------------ comm helpers
    # Thin wrappers with graceful fallbacks for backends that lack the
    # "_tensor" fused ops (e.g. some gloo builds), so the CPU tests can run.

    def _all_gather_flat(self, out, chunk):
        """all_gather_into_tensor(out, chunk); returns async handle or None."""
        if self.world_size == 1:
            return None
        if not self._gloo_no_all_gather_base or dist.get_backend() == "nccl":
            try:
                return dist.all_gather_into_tensor(out, chunk, async_op=True).get_future()
            except RuntimeError:
                if dist.get_backend() != "nccl":
                    self._gloo_no_all_gather_base = True
                else:
                    raise
        # fallback: sync all_gather + cat (chunk_len elements per rank)
        parts = [torch.empty(chunk.numel(), dtype=chunk.dtype, device=chunk.device) for _ in range(self.world_size)]
        dist.all_gather(parts, chunk)
        out.copy_(torch.cat(parts))
        return None

    def _reduce_scatter_avg(self, out, flat):
        """Exact-mean reduce-scatter: SUM the flat buffers across ranks, slice out
        this rank's chunk, then divide by world_size. Returns an async handle whose
        wait() guarantees `out` already holds the exact mean.

        Why not ReduceOp.AVG: the backend is allowed to split the sum over ranks
        arbitrarily (and differently on different ranks); AVG then applies the
        (Ng)/N scaling *inside that partially-summed tree*, which:
        - is NOT bit-exact for non-power-of-2 world sizes (each rank rounds a
          different partial sum), and
        - on 10k-GPU runs drifts chaotically as ranks disagree in the last ulp.
        SUM + one division by the integer N computes the same reduction tree on
        every rank and applies exactly one identical rounding, so all ranks hold
        bit-identical means at ANY world size. SUM runs in the buffers' dtype
        (reduce_dtype: fp32 for fp32 masters, fp64 for fp64 masters), so the sum
        itself is exact to that dtype regardless of how many terms it adds.

        Falls back to sync all_reduce + slice for backends without the fused op.
        Handles both 1-D flat buffers (element-offset math) and dim-0 stacks
        where flat = world_size copies of out stacked along dim 0 (muon).
        """
        if self.world_size == 1:
            out.copy_(flat.reshape(out.shape))
            return None
        if not self._gloo_no_reduce_scatter or dist.get_backend() == "nccl":
            try:
                handle = dist.reduce_scatter_tensor(out, flat, op=dist.ReduceOp.SUM, async_op=True).get_future()
                return _ScaledMean(handle, out, self.world_size)
            except RuntimeError:
                if dist.get_backend() != "nccl":
                    self._gloo_no_reduce_scatter = True
                else:
                    raise
        tmp = flat.clone()
        dist.all_reduce(tmp, op=dist.ReduceOp.SUM)
        if flat.dim() == out.dim() + 1:  # stacked: slice along dim 0 in chunk units
            chunk0 = flat.shape[0] // self.world_size
            out.copy_(tmp[self.rank * chunk0 : (self.rank + 1) * chunk0])
        else:  # 1-D flat buffer: slice by element offsets
            lo, hi = flat_rank_slice(flat.numel(), self.world_size, self.rank)
            out.copy_(tmp.view(-1)[lo:hi].view(out.shape))
        out.div_(self.world_size)
        return None

    # ------------------------------------------------------------ checkpointing

    @torch.no_grad()
    def consolidated_state_dict(self, keep_master_dtype=False):
        """
        Rebuild the standard full model state dict (original dtypes restored).
        Must be called on ALL ranks (it runs collectives); rank0's dict is the
        one passed to save_checkpoint. Keys are identical to model.state_dict().
        keep_master_dtype=True (tests only): every rank keeps the raw fp64/fp32
        master payloads, enabling bitwise cross-rank consistency checks.
        """
        assert self._sharded, "call shard() first"
        sd = {}
        use_dist = self.world_size > 1
        for shard in self._shards_in_order:
            key = self._param_keys[shard.param]
            if shard.kind == "adamw_small":
                sd[key] = shard.param.data.detach().clone()
                continue
            if shard.kind == "muon":
                full = torch.empty(shard.shape, dtype=self.master_dtype, device=self.device)
                if shard.owner == self.rank or not use_dist:
                    full.copy_(shard.master)
                if use_dist:
                    dist.broadcast(full, src=shard.owner)
                payload = full
            else:  # adamw_flat
                g = shard.group
                full_flat = torch.empty(g.padded_numel, dtype=self.master_dtype, device=self.device)
                if use_dist:
                    handle = self._all_gather_flat(full_flat, g.master_chunk)
                    if handle is not None:
                        handle.wait()
                else:
                    full_flat[: g.total_numel] = g.master_chunk
                payload = full_flat[shard.offset : shard.offset + shard.numel].view(shard.shape)
            if self.rank == 0 and not keep_master_dtype:
                sd[key] = payload.to(shard.orig_dtype).contiguous()  # kept for saving
            # non-rank0 payloads are discarded with the dict; don't waste a copy
            else:
                sd[key] = payload
        # Persistent non-trainable buffers (HiPPO A/A_bar/B_bar) are not in
        # optimizer groups but are essential to reconstruct the right brain.
        if self.rank == 0 or keep_master_dtype:
            for key, value in self.model.state_dict().items():
                if key not in sd and key not in self._param_keys.values():
                    sd[key] = value.detach().clone()
        return sd

    # ------------------------------------------------------------------ stats

    def _print_stats(self):
        mb = 1024 * 1024
        mbytes = torch.empty((), dtype=self.master_dtype).element_size()
        master_bytes = 0
        state_bytes = 0
        sharded_params = 0
        for shard in self._shards_in_order:
            if shard.kind == "muon":
                sharded_params += shard.numel
                if self.world_size == 1 or shard.owner == self.rank:
                    master_bytes += shard.numel * mbytes
                    state_bytes += shard.numel * mbytes + shard.numel * mbytes // 4  # momentum + factored 2nd
            elif shard.kind == "adamw_flat":
                sharded_params += shard.numel
                g = shard.group
                master_bytes += g.chunk_len * mbytes
                state_bytes += 2 * g.chunk_len * mbytes
            else:
                master_bytes += shard.numel * 4  # small params stay replicated fp32
                state_bytes += 2 * shard.numel * 4
        print0(
            f"[fsdp] rank {self.rank}/{self.world_size} | master_dtype={self.master_dtype} | "
            f"reduce_dtype={self.reduce_dtype} | "
            f"sharded params: {sharded_params:,} | persistent/rank ≈ "
            f"(weights {master_bytes / mb:.1f} MB + opt state {state_bytes / mb:.1f} MB)"
        )


# -----------------------------------------------------------------------------
# the FSDP-aware optimizer


class FSDPMuonAdamW(MuonAdamW):
    """
    Drop-in replacement for MuonAdamW under FSDP weight sharding.

    Same param_groups, same fused kernels, same ZeRO-2 gradient reduce; two
    differences vs stock:
    - AdamW "large" groups update a flat master *chunk* in place (no param
      all-gather at the end) - weights stay sharded.
    - Muon groups skip the final all-gather-back too: the owning rank updates
      its master directly; non-owners receive fresh weights via broadcast at
      the next forward.
    """

    def __init__(self, param_groups, fsdp: FullySharded):
        super().__init__(param_groups)
        self.fsdp = fsdp
        # adamw param_groups -> AdamWFlatGroup (None = tiny replicated group)
        self._flat_group_of = {}
        first_param_of = {}
        for g in fsdp.adamw_flat_groups:
            first_param_of[id(g.params[0])] = g
        for pg in self.param_groups:
            if pg["kind"] == "adamw":
                g = first_param_of.get(id(pg["params"][0]))
                self._flat_group_of[id(pg)] = g

    # ------------------------------------------------------------------ step

    @torch.no_grad()
    def step(self):
        if dist.is_available() and dist.is_initialized():
            rank, world_size = dist.get_rank(), dist.get_world_size()
        else:
            rank, world_size = 0, 1

        # Phase 1: launch all async reduce ops
        plan = []
        for group in self.param_groups:
            kind = group["kind"]
            if kind == "muon":
                plan.append(("muon", group, self._reduce_muon(group, world_size)))
            elif kind == "adamw":
                fg = self._flat_group_of[id(group)]
                if fg is None:
                    plan.append(("adamw_small", group, self._reduce_adamw(group, world_size)))
                else:
                    plan.append(("adamw_flat", group, self._reduce_adamw_flat(group, fg, world_size)))
            else:
                raise ValueError(f"Unknown optimizer kind: {kind}")

        # Phase 2: wait for reduces, compute updates (no Phase 3 for sharded params:
        # no gathers). Tiny replicated groups may still contain params that took the
        # stock large-param path, so honor any gathers they queued.
        gather_list = []
        for kind, group, info in plan:
            if kind == "muon":
                self._compute_muon_fsdp(group, info, rank, world_size)
            elif kind == "adamw_small":
                self._compute_adamw(group, info, gather_list, rank, world_size)
            else:
                self._compute_adamw_flat(group, self._flat_group_of[id(group)], info)
        if gather_list:
            self._finish_gathers(gather_list)

    # ------------------------------------------------------------- adamw flat

    def _reduce_adamw_flat(self, group, g: AdamWFlatGroup, world_size):
        """Copy the group's full-size grads into the flat padded buffer and
        reduce-scatter (AVG) so this rank holds its chunk's grads."""
        for p in group["params"]:
            assert p.grad is not None, "FSDP adamw group has a param without grad (was its unit's forward run?)"
            offset, numel = g.offset_of[p]
            g.grad_flat[offset : offset + numel].copy_(p.grad.reshape(-1))
        handle = self.fsdp._reduce_scatter_avg(g.grad_chunk, g.grad_flat)
        return dict(future=handle)

    def _compute_adamw_flat(self, group, g: AdamWFlatGroup, info):
        """Fused AdamW on this rank's flat master chunk, in place. No gather."""
        if info["future"] is not None:
            info["future"].wait()
        grad_chunk = g.grad_chunk.to(g.master_dtype)  # upcast the reduced chunk
        g.step += 1
        self._adamw_step_t.fill_(g.step)
        self._adamw_lr_t.fill_(group["lr"])
        self._adamw_beta1_t.fill_(group["betas"][0])
        self._adamw_beta2_t.fill_(group["betas"][1])
        self._adamw_eps_t.fill_(group["eps"])
        self._adamw_wd_t.fill_(group["weight_decay"])
        adamw_step_fused(
            g.master_chunk, grad_chunk, g.exp_avg, g.exp_avg_sq,
            self._adamw_step_t, self._adamw_lr_t, self._adamw_beta1_t,
            self._adamw_beta2_t, self._adamw_eps_t, self._adamw_wd_t,
        )

    # ------------------------------------------------------------- muon (fsdp)

    def _reduce_muon(self, group, world_size):
        """Stock ZeRO-2 muon grad reduce (stack + zero-pad + reduce_scatter), but
        the stack is built in reduce_dtype (>= fp32; fp64 for fp64 masters) so the
        cross-rank SUM is exact even when compute/grads are bf16 - building the
        stack from raw bf16 grads and averaging in bf16 was precision root cause
        #3. Falls back gracefully for backends lacking reduce_scatter_tensor."""
        params = group["params"]
        for p in params:
            assert p.grad is not None, "Muon group has a param without grad"
        reduce_dtype = self.fsdp.reduce_dtype
        if world_size == 1:
            grad_chunk = torch.empty(len(params), *self.fsdp.registry[params[0]].shape,
                                     dtype=reduce_dtype, device=params[0].grad.device)
            for i, p in enumerate(params):
                grad_chunk[i].copy_(p.grad)
            return dict(future=None, grad_chunk=grad_chunk, stacked_grads=None, chunk_size=len(params))
        chunk_size = muon_chunk_size(len(params), world_size)
        padded_num_params = chunk_size * world_size
        meta0 = self.fsdp.registry[params[0]]
        shape, device = meta0.shape, params[0].grad.device
        stacked_grads = torch.empty(padded_num_params, *shape, dtype=reduce_dtype, device=device)
        for i, p in enumerate(params):
            stacked_grads[i].copy_(p.grad)  # upcast bf16/fp32 grads into the reduce dtype
        if len(params) < padded_num_params:
            stacked_grads[len(params):].zero_()
        grad_chunk = torch.empty(chunk_size, *shape, dtype=reduce_dtype, device=device)
        handle = self.fsdp._reduce_scatter_avg(grad_chunk, stacked_grads)
        return dict(future=handle, grad_chunk=grad_chunk, stacked_grads=stacked_grads, chunk_size=chunk_size)

    def _reduce_adamw(self, group: dict, world_size: int) -> dict:
        """Replicated small AdamW groups: exact-mean variant of the stock reduce.
        SUM + one division instead of ReduceOp.AVG - see
        FullySharded._reduce_scatter_avg for why AVG is not bit-exact at
        non-power-of-2 / very large world sizes. Returns the same info dict
        shape as the stock version (its _compute_adamw consumes it unchanged;
        the _ScaledMean wrapper answers the same future.wait())."""
        if world_size == 1:
            return super()._reduce_adamw(group, world_size)
        param_infos = {}
        for p in group['params']:
            grad = p.grad
            if p.numel() < 1024:
                future = dist.all_reduce(grad, op=dist.ReduceOp.SUM, async_op=True).get_future()
                param_infos[p] = dict(future=_ScaledMean(future, grad, world_size), grad_slice=grad, is_small=True)
            else:
                assert dim0_shard_ok(grad.shape[0], world_size), \
                    f"AdamW reduce_scatter requires shape[0] ({grad.shape[0]}) divisible by world_size ({world_size})"
                rank_size = grad.shape[0] // world_size
                grad_slice = torch.empty_like(grad[:rank_size])
                future = dist.reduce_scatter_tensor(grad_slice, grad, op=dist.ReduceOp.SUM, async_op=True).get_future()
                param_infos[p] = dict(future=_ScaledMean(future, grad_slice, world_size), grad_slice=grad_slice, is_small=False)
        return dict(param_infos=param_infos)

    def _compute_muon_fsdp(self, group, info, rank, world_size):
        """Stock Muon update on the params this rank owns; the update is written
        straight into the owner's master tensors. The stock all-gather-back is
        intentionally absent: non-owner ranks get fresh weights via broadcast at
        the next forward."""
        if info["future"] is not None:
            info["future"].wait()
        params = group["params"]
        meta0 = self.fsdp.registry[params[0]]
        shape = meta0.shape
        device = params[0].grad.device
        master_dtype = self.fsdp.master_dtype
        chunk_size = info["chunk_size"]
        grad_chunk = info["grad_chunk"].to(master_dtype)  # upcast the reduced grads
        start_idx, num_owned = muon_owned_range(len(params), world_size, rank)

        state = self.state[params[0]]
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(chunk_size, *shape, dtype=master_dtype, device=device)
        if "second_momentum_buffer" not in state:
            state_shape = ((chunk_size, *shape[:-2], shape[-2], 1) if shape[-2] >= shape[-1]
                           else (chunk_size, *shape[:-2], 1, shape[-1]))
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=master_dtype, device=device)
        red_dim = -1 if shape[-2] >= shape[-1] else -2

        if num_owned > 0:
            owned = [params[start_idx + i] for i in range(num_owned)]
            stacked_owned = torch.stack(owned)  # owners' param.data are the fp32/fp64 masters here
            self._muon_momentum_t.fill_(group["momentum"])
            self._muon_beta2_t.fill_(group["beta2"])
            self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1]) ** 0.5)
            self._muon_wd_t.fill_(group["weight_decay"])
            muon_step_fused(
                grad_chunk[:num_owned], stacked_owned,
                state["momentum_buffer"][:num_owned], state["second_momentum_buffer"][:num_owned],
                self._muon_momentum_t, self._muon_lr_t, self._muon_wd_t, self._muon_beta2_t,
                group["ns_steps"], red_dim,
            )
            # write the updates back into the owner's masters (pure local copy, no comm)
            for i, p in enumerate(owned):
                p.data.copy_(stacked_owned[i])

    # ---------------------------------------------------------- state save/load
    # Optimizer state layout under FSDP:
    # - adamw flat groups: one entry (step, exp_avg, exp_avg_sq chunk tensors)
    #   stored under the group's FIRST param index;
    # - muon / small groups: stock per-param entries (keyed at the group's first
    #   param for muon), identical to stock MuonAdamW.
    # Resuming requires the same world_size (chunks are rank-shaped), matching
    # stock FoxSay's rank-sharded optimizer checkpoint convention.

    def _param_index_map(self):
        mapping = {}
        i = 0
        for group in self.param_groups:
            for p in group["params"]:
                mapping[p] = i
                i += 1
        return mapping

    def state_dict(self):
        mapping = self._param_index_map()
        state = {}
        for g in self.fsdp.adamw_flat_groups:
            state[mapping[g.params[0]]] = dict(step=g.step, exp_avg=g.exp_avg, exp_avg_sq=g.exp_avg_sq)
        for p, s in self.state.items():
            idx = mapping[p]
            if idx not in state:
                state[idx] = {k: v for k, v in s.items()}
        param_groups = [{k: v for k, v in g.items() if k != "params"} for g in self.param_groups]
        return {"state": state, "param_groups": param_groups}

    def load_state_dict(self, state_dict):
        mapping = self._param_index_map()
        reverse = {i: p for p, i in mapping.items()}
        # 1) group hyperparams
        assert len(state_dict["param_groups"]) == len(self.param_groups), "param_groups count mismatch"
        for g_saved, g_cur in zip(state_dict["param_groups"], self.param_groups):
            for k, v in g_saved.items():
                if k != "params":
                    g_cur[k] = v
        # 2) flat group states
        flat_by_first_idx = {mapping[g.params[0]]: g for g in self.fsdp.adamw_flat_groups}
        for idx_str, s_saved in state_dict["state"].items():
            idx = int(idx_str)
            if idx in flat_by_first_idx:
                g = flat_by_first_idx[idx]
                assert s_saved["exp_avg"].numel() == g.chunk_len, (
                    f"FSDP optimizer checkpoint was saved with a different world_size "
                    f"(chunk {s_saved['exp_avg'].numel()} != current {g.chunk_len}). "
                    f"Resume with the same GPU count."
                )
                g.step = int(s_saved["step"])
                g.exp_avg.copy_(s_saved["exp_avg"])
                g.exp_avg_sq.copy_(s_saved["exp_avg_sq"])
            else:
                p = reverse[idx]
                self.state[p] = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in s_saved.items()}
