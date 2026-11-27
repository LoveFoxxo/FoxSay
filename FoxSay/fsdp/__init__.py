"""
FSDP (ZeRO-3 style) weight sharding for FoxSay.

Import lazily: FoxSay/optim.py imports FoxSay.fsdp.shard_math at load time,
so this package's __init__ must not eagerly import core.py (which imports
FoxSay.optim back). Both import styles below therefore work:

    from FoxSay.fsdp import FullySharded, FSDPMuonAdamW
    from FoxSay.fsdp.core import FullySharded
"""

__all__ = ["FullySharded", "FSDPMuonAdamW"]


def __getattr__(name):
    if name in ("FullySharded", "FSDPMuonAdamW"):
        from FoxSay.fsdp import core
        return getattr(core, name)
    raise AttributeError(f"module 'FoxSay.fsdp' has no attribute {name!r}")
