import json

from src.prompting import format_tools, build_prompt_find_function


def test_format_tools_returns_valid_json_string() -> None:
    tools = {"fn_is_even": {"n": {"type": "integer"}}}
    result = format_tools(tools)

    assert json.loads(result) == tools


def test_build_prompt_find_function_includes_user_prompt() -> None:
    prompt = build_prompt_find_function("Is 4 an even number?", "{}")

    assert "Is 4 an even number?" in prompt


def test_build_prompt_find_function_includes_fn_none_fallback() -> None:
    prompt = build_prompt_find_function("hello", "{}")

    assert "fn_none" in prompt


def test_build_prompt_find_function_ends_with_assistant_prefix() -> None:
    prompt = build_prompt_find_function("hello", "{}")

    assert prompt.endswith('Assistant: {"name": "')
