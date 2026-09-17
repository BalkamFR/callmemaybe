import json
import sys
from pathlib import Path

import pytest

from src.parsing import Config


def test_config_defaults() -> None:
    config = Config()

    default_functions = "data/input/functions_definition.json"
    assert config.functions_definition == default_functions
    assert config.input == "data/input/function_calling_tests.json"
    assert config.output == "data/output/function_calling_results.json"
    assert config.model == "Qwen/Qwen3-0.6B"


def test_parse_arguments_overrides_defaults(
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", [
        "prog",
        "--input", "custom_input.json",
        "--output", "custom_output.json",
        "--model", "Qwen/Qwen2.5-0.5B",
    ])

    config = Config.parse_arguments()

    assert config.input == "custom_input.json"
    assert config.output == "custom_output.json"
    assert config.model == "Qwen/Qwen2.5-0.5B"
    default_functions = "data/input/functions_definition.json"
    assert config.functions_definition == default_functions


def test_parse_arguments_no_args_keeps_defaults(
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["prog"])

    config = Config.parse_arguments()

    assert config == Config()


def test_load_tools_reads_functions_definition(tmp_path: Path) -> None:
    functions_file = tmp_path / "functions_definition.json"
    functions_file.write_text(json.dumps([
        {"name": "fn_is_even", "parameters": {"n": {"type": "integer"}}}
    ]))
    config = Config(functions_definition=str(functions_file))

    tools = config.load_tools()

    assert tools == {"fn_is_even": {"n": {"type": "integer"}}}


def test_load_tools_missing_key_raises_clear_error(tmp_path: Path) -> None:
    functions_file = tmp_path / "functions_definition.json"
    functions_file.write_text(json.dumps([{"description": "no name field"}]))
    config = Config(functions_definition=str(functions_file))

    try:
        config.load_tools()
        assert False, "expected ValueError"
    except ValueError as e:
        assert "name" in str(e)


def test_load_prompt_reads_prompts(tmp_path: Path) -> None:
    input_file = tmp_path / "function_calling_tests.json"
    input_file.write_text(json.dumps([{"prompt": "Is 4 an even number?"}]))
    config = Config(input=str(input_file))

    prompts = config.load_prompt()

    assert prompts == ["Is 4 an even number?"]
