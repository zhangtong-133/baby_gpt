import numpy as np

from baby_gpt.model import BigramLanguageModel


def test_bigram_training_step_reduces_repeated_pair_loss() -> None:
    rng = np.random.default_rng(0)
    model = BigramLanguageModel(vocab_size=2, rng=rng)
    x = np.zeros(64, dtype=np.int64)
    y = np.ones(64, dtype=np.int64)

    before = model.loss(x, y)
    for _ in range(20):
        model.train_step(x, y, learning_rate=0.5)
    after = model.loss(x, y)

    assert after < before


def test_bigram_generate_shape() -> None:
    rng = np.random.default_rng(1)
    model = BigramLanguageModel(vocab_size=3, rng=rng)
    ids = model.generate(start_id=0, max_new_tokens=5, rng=rng)
    assert len(ids) == 6
    assert all(0 <= idx < 3 for idx in ids)
