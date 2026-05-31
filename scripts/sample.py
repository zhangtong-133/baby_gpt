import argparse

import _bootstrap  # noqa: F401
import numpy as np

from baby_gpt.config import BigramConfig
from baby_gpt.train import train_bigram


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="语")
    parser.add_argument("--tokens", type=int, default=120)
    args = parser.parse_args()

    config = BigramConfig(sample_tokens=args.tokens)
    model, tokenizer = train_bigram(config)
    start = args.prompt[-1]
    start_id = tokenizer.encode(start)[0]
    ids = model.generate(start_id, config.sample_tokens, np.random.default_rng(config.seed + 2))
    print()
    print(args.prompt[:-1] + tokenizer.decode(ids))


if __name__ == "__main__":
    main()
