import json
from pathlib import Path
from typing import Any

from src.utils import (
    convert_float, is_path_syntax, clean_json_output, create_json
)


def test_convert_float_adds_decimal_to_integers() -> None:
    assert convert_float('"a": 3, "b": 5') == '"a": 3.0, "b": 5.0'


def test_convert_float_preserves_negative_integers() -> None:
    assert convert_float('"n": -4') == '"n": -4.0'


def test_convert_float_does_not_touch_existing_floats() -> None:
    assert convert_float('"rate": 0.0375') == '"rate": 0.0375'


def test_is_path_syntax_detects_unix_path() -> None:
    assert is_path_syntax("/home/user/data.json") is True


def test_is_path_syntax_detects_windows_path() -> None:
    assert is_path_syntax("C:\\Users\\john\\config.ini") is True


def test_is_path_syntax_rejects_plain_string() -> None:
    assert is_path_syntax("SELECT * FROM users") is False


def test_is_path_syntax_rejects_non_string() -> None:
    non_string: Any = 42
    assert is_path_syntax(non_string) is False


def test_is_path_syntax_rejects_empty_string() -> None:
    assert is_path_syntax("   ") is False


def test_clean_json_output_removes_trailing_comma() -> None:
    raw = '{"name": "fn_is_even", "parameters": {"n": 4,}}'
    result = clean_json_output(raw)
    assert result == '{"name": "fn_is_even", "parameters": {"n": 4}}'


def test_clean_json_output_extracts_first_json_object() -> None:
    raw = (
        '{"name": "fn_is_even", "parameters": {"n": 4}}'
        '\nAssistant: {"garbage"'
    )
    result = clean_json_output(raw)
    assert result == '{"name": "fn_is_even", "parameters": {"n": 4}}'


def test_clean_json_output_strips_path_values() -> None:
    raw = (
        '{"name": "fn_read_file", '
        '"parameters": {"path": " /home/user/data.json"}}'
    )
    result = clean_json_output(raw)
    assert '"path": "/home/user/data.json"' in result


def test_create_json_writes_valid_json_file(tmp_path: Path) -> None:
    output_path = tmp_path / "out" / "result.json"
    entry = {"prompt": "hi", "name": "fn_none", "parameters": {}}
    entries: list[Any] = [entry]
    create_json(entries, str(output_path))

    assert output_path.exists()
    data = json.loads(output_path.read_text())
    assert data == [{"prompt": "hi", "name": "fn_none", "parameters": {}}]
