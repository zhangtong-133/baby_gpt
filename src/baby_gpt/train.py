import numpy as np

from baby_gpt.config import BigramConfig
from baby_gpt.dataset import encode_file, random_bigram_batch, train_val_split
from baby_gpt.model import BigramLanguageModel


def train_bigram(config: BigramConfig) -> tuple[BigramLanguageModel, object]:
    rng = np.random.default_rng(config.seed)
    ids, tokenizer = encode_file(config.data_path)
    train_ids, val_ids = train_val_split(ids)
    model = BigramLanguageModel(tokenizer.vocab_size, rng)

    for step in range(1, config.steps + 1):
        x, y = random_bigram_batch(train_ids, config.batch_size, rng)
        train_loss = model.train_step(x, y, config.learning_rate)

        if step == 1 or step % config.eval_interval == 0:
            vx, vy = random_bigram_batch(val_ids, min(config.batch_size, len(val_ids) - 1), rng)
            val_loss = model.loss(vx, vy)
            print(f"step {step:04d} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

    return model, tokenizer
