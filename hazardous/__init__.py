from pathlib import Path

from ._bagging import BaggingSurvival
from ._survival_boost import SurvivalBoost

with open(Path(__file__).parent / "VERSION.txt") as _fh:
    __version__ = _fh.read().strip()


__all__ = ["metrics", "SurvivalBoost", "BaggingSurvival", "calibration"]
