from pathlib import Path

from ._gradient_boosting_incidence import GradientBoostingIncidence
from ._ipcw import IPCWEstimator

with open(Path(__file__).parent / "VERSION.txt") as _fh:
    __version__ = _fh.read().strip()


__all__ = [
    "metrics",
    "GradientBoostingIncidence",
    "IPCWEstimator",
]
