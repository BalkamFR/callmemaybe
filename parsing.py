import json
from pathlib import Path
from typing import Any


def load_tools(
        path: str | Path = "data/input/functions_definition.json") -> dict[str, dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        tools: list[dict[str, Any]] = json.load(f)

    return {fn["name"]: fn["parameters"] for fn in tools}


def load_prompt(
        path: str | Path = "data/input/function_calling_tests.json") -> dict[str, dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        tools: list[dict[str, Any]] = json.load(f)

    return {fn["prompt"] for fn in tools}


