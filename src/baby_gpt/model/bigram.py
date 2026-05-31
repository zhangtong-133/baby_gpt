import numpy as np


class BigramLanguageModel:
    """A trainable bigram language model with logits[vocab, vocab]."""

    def __init__(self, vocab_size: int, rng: np.random.Generator):
        self.vocab_size = vocab_size
        self.logits = 0.01 * rng.standard_normal((vocab_size, vocab_size))

    def probabilities(self, x: np.ndarray) -> np.ndarray:
        logits = self.logits[x]
        shifted = logits - logits.max(axis=-1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=-1, keepdims=True)

    def loss(self, x: np.ndarray, y: np.ndarray) -> float:
        probs = self.probabilities(x)
        n = len(x)
        return float(-np.log(probs[np.arange(n), y] + 1e-12).mean())

    def train_step(self, x: np.ndarray, y: np.ndarray, learning_rate: float) -> float:
        probs = self.probabilities(x)
        n = len(x)
        loss = float(-np.log(probs[np.arange(n), y] + 1e-12).mean())

        grad = probs
        grad[np.arange(n), y] -= 1.0
        grad /= n

        dlogits = np.zeros_like(self.logits)
        np.add.at(dlogits, x, grad)
        self.logits -= learning_rate * dlogits
        return loss

    def generate(self, start_id: int, max_new_tokens: int, rng: np.random.Generator) -> list[int]:
        ids = [start_id]
        current = start_id
        for _ in range(max_new_tokens):
            probs = self.probabilities(np.array([current]))[0]
            current = int(rng.choice(self.vocab_size, p=probs))
            ids.append(current)
        return ids
