from src.utils import parse_n_split


def test_parse_n_split_accepts_bracketed_count():
    assert parse_n_split("n[50]") == 50


def test_parse_n_split_ignores_named_splits():
    assert parse_n_split("mini") is None
    assert parse_n_split("test") is None


def test_parse_n_split_rejects_malformed_counts():
    assert parse_n_split("n50") is None
    try:
        parse_n_split("n[]")
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for malformed n split")
