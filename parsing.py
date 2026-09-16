import json
from pathlib import Path
from typing import Any


def load_tools(
        path: str | Path = "data/input/functions_definition.json") -> dict[str, dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        tools: list[dict[str, Any]] = json.load(f)

    return {fn["name"]: fn["parameters"] for fn in tools}


def load_prompt(
        path: str | Path = "data/input/function_calling_tests.json") -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        tools: list[dict[str, Any]] = json.load(f)

    return [fn["prompt"] for fn in tools]


import sys

import sys

def parse_arguments() -> dict:
    config = {
        "functions_definition": "data/input/functions_definition.json",
        "input": "data/input/function_calling_tests.json",
        "output": "data/output/function_calling_results.json"
    }
    
    args = sys.argv[1:]
    
    for i in range(len(args) - 1):
        if args[i] == "--functions_definition":
            config["functions_definition"] = args[i + 1]
        elif args[i] == "--input":
            config["input"] = args[i + 1]
        elif args[i] == "--output":
            config["output"] = args[i + 1]
            
    return config