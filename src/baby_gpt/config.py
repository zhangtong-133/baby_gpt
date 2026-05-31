from dataclasses import dataclass


@dataclass(frozen=True)
class BigramConfig:
    data_path: str = "data/raw/tiny.txt"
    seed: int = 42
    steps: int = 500
    batch_size: int = 32
    learning_rate: float = 0.8
    eval_interval: int = 100
    sample_tokens: int = 120
