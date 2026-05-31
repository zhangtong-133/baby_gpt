import _bootstrap  # noqa: F401

from baby_gpt.config import BigramConfig
from baby_gpt.train import train_bigram


def main() -> None:
    config = BigramConfig()
    model, tokenizer = train_bigram(config)
    start_id = tokenizer.encode("语")[0] if "语" in tokenizer.chars else 0
    ids = model.generate(start_id, config.sample_tokens, __import__("numpy").random.default_rng(config.seed + 1))
    print()
    print(tokenizer.decode(ids))


if __name__ == "__main__":
    main()
