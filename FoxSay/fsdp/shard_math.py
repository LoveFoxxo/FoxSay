"""
Pure, dependency-free sharding math shared by the ZeRO-2 optimizer (FoxSay/optim.py)
and the FSDP weight-sharding layer (FoxSay/fsdp/core.py).

Kept free of torch/distributed imports on purpose: this module is imported at load
time by FoxSay.optim, so it must never pull in the rest of the fsdp package.
"""


def ceil_div(a: int, b: int) -> int:
    """Integer ceil(a / b)."""
    return -(-a // b)


def muon_chunk_size(num_params: int, world_size: int) -> int:
    """Number of whole parameters each rank owns inside a Muon shape-group.

    Mirrors the reduce_scatter chunking in optim.MuonAdamW._reduce_muon: K params
    are zero-padded to ceil(K/N)*N and scattered, so rank r owns params
    [r*chunk_size, (r+1)*chunk_size).
    """
    return ceil_div(num_params, world_size)


def muon_owned_range(num_params: int, world_size: int, rank: int) -> tuple[int, int]:
    """(start_idx, num_owned) of the params rank `rank` owns in a Muon group."""
    chunk_size = muon_chunk_size(num_params, world_size)
    start_idx = rank * chunk_size
    num_owned = min(chunk_size, max(0, num_params - start_idx))
    return start_idx, num_owned


def flat_chunk_bounds(total_numel: int, world_size: int) -> tuple[int, int]:
    """(chunk_len, padded_numel) for evenly flat-chunking a buffer of total_numel
    elements across world_size ranks. Any world size works (unlike dim-0 slicing,
    which requires divisibility) — this is what makes 100-GPU runs possible for
    vocab-sized tensors like 32768 x n_embd."""
    chunk_len = ceil_div(total_numel, world_size)
    padded_numel = chunk_len * world_size
    return chunk_len, padded_numel


def flat_rank_slice(total_numel: int, world_size: int, rank: int) -> tuple[int, int]:
    """(lo, hi) range of the padded flat buffer owned by `rank`."""
    chunk_len, padded_numel = flat_chunk_bounds(total_numel, world_size)
    lo = rank * chunk_len
    hi = min(lo + chunk_len, padded_numel)
    return lo, hi


def dim0_shard_ok(dim0: int, world_size: int) -> bool:
    """Whether a tensor can be reduce_scattered along dim 0 (the ZeRO-2 path)."""
    return dim0 % world_size == 0
