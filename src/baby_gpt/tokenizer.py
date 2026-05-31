from dataclasses import dataclass


@dataclass(frozen=True)
class CharTokenizer:
    chars: tuple[str, ...]

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        chars = tuple(sorted(set(text)))
        if not chars:
            raise ValueError("Cannot build a tokenizer from empty text.")
        return cls(chars=chars)

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    @property
    def stoi(self) -> dict[str, int]:
        return {ch: idx for idx, ch in enumerate(self.chars)}

    @property
    def itos(self) -> dict[int, str]:
        return {idx: ch for idx, ch in enumerate(self.chars)}

    def encode(self, text: str) -> list[int]:
        stoi = self.stoi
        try:
            return [stoi[ch] for ch in text]
        except KeyError as exc:
            raise ValueError(f"Unknown character: {exc.args[0]!r}") from exc

    def decode(self, ids: list[int]) -> str:
        itos = self.itos
        try:
            return "".join(itos[idx] for idx in ids)
        except KeyError as exc:
            raise ValueError(f"Unknown token id: {exc.args[0]!r}") from exc
