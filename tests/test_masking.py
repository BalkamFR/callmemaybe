from src.masking import mask_logits, get_allowed_next_tokens


def test_mask_logits_keeps_only_allowed_tokens() -> None:
    logits = [1.0, 2.0, 3.0, 4.0]
    result = mask_logits(logits, [1, 3])

    assert result[0] == float("-inf")
    assert result[1] == 2.0
    assert result[2] == float("-inf")
    assert result[3] == 4.0


def test_mask_logits_empty_allowed_returns_all_masked() -> None:
    logits = [1.0, 2.0, 3.0]
    result = mask_logits(logits, [])

    assert result == [float("-inf")] * 3


def test_get_allowed_next_tokens_returns_next_token_for_prefix() -> None:
    encodings = [[1, 2, 3], [1, 5, 6]]
    result = get_allowed_next_tokens(encodings, [1])

    assert set(result) == {2, 5}


def test_get_allowed_next_tokens_excludes_completed_candidates() -> None:
    encodings = [[1, 2]]
    result = get_allowed_next_tokens(encodings, [1, 2])

    assert result == []


def test_get_allowed_next_tokens_no_candidates_match() -> None:
    encodings = [[1, 2, 3]]
    result = get_allowed_next_tokens(encodings, [9])

    assert result == []


def test_get_allowed_next_tokens_deduplicates() -> None:
    encodings = [[1, 2], [1, 2, 3]]
    result = get_allowed_next_tokens(encodings, [1])

    assert result == [2]
