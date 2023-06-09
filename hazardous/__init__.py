from pathlib import Path

with open(Path(__file__).parent / "VERSION.txt") as _fh:
    __version__ = _fh.read().strip()
