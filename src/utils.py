from pathlib import Path
from typing import Any
import re
import json

from llm_sdk import Small_LLM_Model


def encoding_fonction(
        model: Small_LLM_Model, text_to_encode: str) -> list[int]:
    """Encode a piece of text into a plain list of token ids."""
    new: list[int] = model.encode(text_to_encode).tolist()[0]
    return new


def convert_float(params_json: str) -> str:
    """Append ``.0`` to bare integers found after a JSON ``:``.

    The model generates whole numbers (e.g. ``3``) for ``number``
    parameters; this turns them into valid floats (``3.0``) without
    touching values that already have a decimal point.
    """
    return re.sub(r"(:\s*)(-?\d+)(?![.\d])", r"\1\2.0", params_json)


def is_path_syntax(value: str) -> bool:
    """Guess whether a string looks like a filesystem path.

    Used to strip stray whitespace the model may have generated
    around path-like values without touching other strings.
    """
    if not isinstance(value, str) or not value.strip():
        return False
    cleaned_value = value.strip()
    path_obj = Path(cleaned_value)

    has_separator = "/" in cleaned_value or "\\" in cleaned_value
    has_suffix = path_obj.suffix != ""
    has_parent = path_obj.parent != Path(".")
    if has_separator or has_suffix or has_parent:
        return True

    return False


def clean_json_output(raw_text: str) -> str:
    """Extract and normalize the JSON tool call from raw model output.

    Removes trailing commas, extracts the first balanced ``{...}``
    object (dropping any text generated after it), and strips
    stray whitespace from path-like parameter values.

    Args:
        raw_text: The raw text produced by the generation loop.

    Returns:
        A JSON string for the tool call, or the best-effort cleaned
        text if it could not be parsed as JSON.
    """
    cleaned = re.sub(r",\s*([\]}])", r"\1", raw_text)
    cleaned = re.sub(r",(\s*,)+", ", ", cleaned)

    start = cleaned.find('{"name":')
    if start == -1:
        start = cleaned.find("{")
    if start == -1:
        return cleaned.strip()

    depth = 0
    in_string = False
    escape = False
    end = len(cleaned)

    for i in range(start, len(cleaned)):
        char = cleaned[i]

        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
        else:
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break

    extracted_json = cleaned[start:end].strip()

    try:
        data = json.loads(extracted_json)
        if "parameters" in data and isinstance(data["parameters"], dict):
            for key, val in data["parameters"].items():
                if is_path_syntax(val):
                    data["parameters"][key] = val.strip()
        return json.dumps(data)
    except Exception:
        return extracted_json


def create_json(
        raw_list: list[Any],
        path: str = "data/output/function_calling_results.json") -> None:
    """Write a list of results to a JSON file, creating parents dirs.

    Args:
        raw_list: Result entries, either already-parsed dicts or
            JSON-encoded strings.
        path: Destination file path.
    """
    parsed_data = [
        item if isinstance(item, dict) else json.loads(item)
        for item in raw_list
    ]
    json_str = json.dumps(parsed_data, indent=4)

    file_path = Path(path)

    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w") as f:
        f.write(json_str)
