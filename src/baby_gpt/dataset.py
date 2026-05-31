from pathlib import Path

import numpy as np

from baby_gpt.tokenizer import CharTokenizer


def load_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def encode_file(path: str | Path) -> tuple[np.ndarray, CharTokenizer]:
    text = load_text(path)
    tokenizer = CharTokenizer.from_text(text)
    ids = np.array(tokenizer.encode(text), dtype=np.int64)
    return ids, tokenizer


def train_val_split(ids: np.ndarray, val_fraction: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    if ids.ndim != 1:
        raise ValueError("Expected a flat token id array.")
    if len(ids) < 2:
        raise ValueError("Need at least two token ids.")

    split = max(1, int(len(ids) * (1.0 - val_fraction)))
    split = min(split, len(ids) - 1)
    return ids[:split], ids[split:]


def random_bigram_batch(ids: np.ndarray, batch_size: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    if len(ids) < 2:
        raise ValueError("Need at least two token ids for bigram training.")
    starts = rng.integers(0, len(ids) - 1, size=batch_size)
    x = ids[starts]
    y = ids[starts + 1]
    return x, y
