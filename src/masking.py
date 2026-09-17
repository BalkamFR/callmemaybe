def mask_logits(
        all_logits: list[float],
        allowed_tokens: list[int]) -> list[float]:
    """Set every logit outside ``allowed_tokens`` to negative infinity.

    Args:
        all_logits: Raw logits for every token in the vocabulary.
        allowed_tokens: Token ids that are allowed to be selected.

    Returns:
        A list the same length as ``all_logits`` where disallowed
        positions are set to negative infinity so they are never
        chosen by argmax.
    """
    masked_logits = [-float("inf")] * len(all_logits)
    for token_id in allowed_tokens:
        masked_logits[token_id] = all_logits[token_id]
    return masked_logits


def get_allowed_next_tokens(
        all_encodings: list[list[int]],
        current_tokens: list[int]) -> list[int]:
    """Find which next token keeps at least one candidate valid.

    Args:
        all_encodings: Token id sequences for every candidate string
            (e.g. every tool name), the full vocabulary of choices.
        current_tokens: Tokens generated so far for the current choice.

    Returns:
        The set of token ids that extend ``current_tokens`` into a
        valid prefix of at least one entry in ``all_encodings``.
    """
    allowed_tokens: list[int] = []
    generated_count = len(current_tokens)

    for candidate_tokens in all_encodings:
        candidate_length = len(candidate_tokens)
        if candidate_length <= generated_count:
            continue
        is_matching = True
        for i in range(generated_count):
            if candidate_tokens[i] != current_tokens[i]:
                is_matching = False
                break
        if not is_matching:
            continue
        next_token = candidate_tokens[generated_count]

        if next_token not in allowed_tokens:
            allowed_tokens.append(next_token)
    return allowed_tokens
