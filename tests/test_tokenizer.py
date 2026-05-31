import pytest

from baby_gpt.tokenizer import CharTokenizer


def test_char_tokenizer_round_trip() -> None:
    tokenizer = CharTokenizer.from_text("abbccc")
    ids = tokenizer.encode("cab")
    assert tokenizer.decode(ids) == "cab"


def test_char_tokenizer_rejects_unknown_character() -> None:
    tokenizer = CharTokenizer.from_text("abc")
    with pytest.raises(ValueError):
        tokenizer.encode("d")
