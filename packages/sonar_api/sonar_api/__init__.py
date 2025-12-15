import pathlib

with open(pathlib.Path(__file__).parent / "VERSION") as f:
    __version__ = f.read().strip()
