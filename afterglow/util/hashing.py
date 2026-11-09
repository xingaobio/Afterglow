"""File and perceptual hashing (ARCHITECTURE.md §1 util/hashing.py)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    """Hex sha256 of a whole file, read in `chunk`-byte blocks."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def dhash(img: Image.Image) -> str:
    """64-bit difference hash as 16 hex characters (9x8 grey, compare horizontal neighbours)."""
    grey = img.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    px = np.asarray(grey, dtype=np.int16)
    diff = (px[:, :-1] > px[:, 1:]).flatten()
    bits = 0
    for bit in diff:
        bits = (bits << 1) | int(bit)
    return f"{bits:016x}"


def hamming(a_hex: str, b_hex: str) -> int:
    """Number of differing bits between two hex hashes."""
    return (int(a_hex, 16) ^ int(b_hex, 16)).bit_count()
