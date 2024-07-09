from pathlib import Path

from ._ipcw import IPCWEstimator

with open(Path(__file__).parent / "VERSION.txt") as _fh:
    __version__ = _fh.read().strip()


__all__ = [
    "metrics",
    "IPCWEstimator",
]
