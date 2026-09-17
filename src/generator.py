from typing import Any
from llm_sdk import Small_LLM_Model
from src.masking import get_allowed_next_tokens, mask_logits
import numpy as np
from src.prompting import build_prompt_find_function, format_tools
from src.utils import encoding_fonction, convert_float, clean_json_output
import re


def get_all_tool_encodings(
        model: Small_LLM_Model, tools: dict[str, Any]) -> list[list[int]]:
    """Pre-encode every tool name to token ids for constrained decoding.

    Args:
        model: The loaded LLM used to tokenize.
        tools: Mapping from tool name to its parameter schema.

    Returns:
        One token id sequence per tool name.
    """
    all_name_encoding = []
    for to in tools:
        all_name_encoding.append(encoding_fonction(model, to))
    return all_name_encoding


def all_numbers_tokens(model: Small_LLM_Model) -> list[int]:
    """List the token ids allowed while generating a numeric value.

    Covers digits 0-9, the decimal point, and both spellings of the
    minus sign (with and without a leading space), since the
    tokenizer merges a preceding space into the sign for negatives.
    """
    new = []
    new.append(model.encode(".")[0].tolist()[0])
    new.append(model.encode("-")[0].tolist()[0])
    new.append(model.encode(" -")[0].tolist()[0])
    for a in range(10):
        new.append(model.encode(str(a))[0].tolist()[0])
    return new


def select_tool_name(
        user_prompt: str,
        model: Small_LLM_Model,
        fonction_allow: list[list[int]],
        tools: dict[str, Any]) -> tuple[str, list[int]]:
    """Generate the tool name for a prompt using constrained decoding.

    At each step, logits are masked so only tokens that extend a
    valid tool name (or ``fn_none``) can be selected, guaranteeing
    the result is always one of the allowed names.

    Args:
        user_prompt: The natural-language request from the user.
        model: The loaded LLM.
        fonction_allow: Pre-encoded token sequences for every tool.
        tools: Mapping from tool name to its parameter schema.

    Returns:
        A tuple of the selected tool name and the full token id
        history so far, to be continued by ``generate_parameters``.
    """
    estimated_name_calls = max(len(enc) for enc in fonction_allow)

    function_tokens: list[int] = []
    local_allow = list(fonction_allow)
    none_encoding = encoding_fonction(model, "fn_none")
    if none_encoding not in local_allow:
        local_allow.append(none_encoding)

    tools_text = format_tools(tools)
    tensor_ids = model.encode(
        build_prompt_find_function(user_prompt, tools_text)
    )
    current_ids = tensor_ids[0].tolist()

    for _ in range(estimated_name_calls):
        logits = model.get_logits_from_input_ids(current_ids)
        allowed_tokens = get_allowed_next_tokens(local_allow, function_tokens)
        if not allowed_tokens:
            break
        final_logit = mask_logits(logits, allowed_tokens)
        token_id = int(np.argmax(final_logit))
        current_ids.append(token_id)
        function_tokens.append(token_id)

    tool_name = model.decode(function_tokens)
    return tool_name, current_ids


def generate_parameters(
        model: Small_LLM_Model,
        current_ids: list[int],
        properties: dict[str, Any],
        calls_per_number_param: int,
        calls_per_string_param: int) -> tuple[str, str]:
    """Generate every parameter value for the already-chosen tool.

    Number parameters are constrained to digits, a sign and a
    decimal point. String parameters are generated freely and then
    truncated at the first unescaped closing quote, discarding
    anything the model produced beyond the actual value.

    Args:
        model: The loaded LLM.
        current_ids: Token id history to continue generating from.
        properties: Parameter schema for the chosen tool.
        calls_per_number_param: Max generation steps per number.
        calls_per_string_param: Max generation steps per string.

    Returns:
        A tuple of the raw ``parameters`` JSON body (without the
        surrounding braces) and the last parameter's name, used by
        the caller to decide whether to run float conversion.
    """
    number_tokens = all_numbers_tokens(model)
    param_keys = list(properties.keys())
    current_ids.extend(encoding_fonction(model, '", "parameters": {'))
    params_start_idx = len(current_ids)
    for idx, param_name in enumerate(param_keys):
        if properties[param_name]["type"] in ("number", "integer"):
            current_ids.extend(encoding_fonction(model, f'"{param_name}":'))
            stop_char = "," if idx != len(param_keys) - 1 else "}"
            stop_token_id = encoding_fonction(model, stop_char)[0]
            space_token_id = encoding_fonction(model, " ")[0]
            allowed_tokens = number_tokens + [stop_token_id, space_token_id]
            for _ in range(calls_per_number_param):
                logits = model.get_logits_from_input_ids(current_ids)
                final_logit = mask_logits(logits, allowed_tokens)
                token_id = int(np.argmax(final_logit))
                current_ids.append(token_id)
                if token_id == stop_token_id:
                    break
        elif properties[param_name]["type"] == "boolean":
            current_ids.extend(encoding_fonction(model, f'"{param_name}": '))
            true_token_id = encoding_fonction(model, "true")[0]
            false_token_id = encoding_fonction(model, "false")[0]
            logits = model.get_logits_from_input_ids(current_ids)
            final_logit = mask_logits(logits, [true_token_id, false_token_id])
            token_id = int(np.argmax(final_logit))
            current_ids.append(token_id)
            if idx != len(param_keys) - 1:
                current_ids.extend(encoding_fonction(model, ", "))
            else:
                current_ids.extend(encoding_fonction(model, "}"))
        elif properties[param_name]["type"] == "string":
            current_ids.extend(encoding_fonction(model, f'"{param_name}": '))
            current_ids.extend(encoding_fonction(model, '"'))
            string_start = len(current_ids)
            for _ in range(calls_per_string_param):
                logits = model.get_logits_from_input_ids(current_ids)
                token_id = int(np.argmax(logits))
                current_ids.append(token_id)
                text_so_far = model.decode(current_ids[string_start:])
                match = re.search(r'(?<!\\)"', text_so_far)
                if match:
                    value_text = text_so_far[:match.start()]
                    current_ids = current_ids[:string_start]
                    current_ids.extend(
                        encoding_fonction(model, value_text + '"')
                    )
                    break
            if idx != len(param_keys) - 1:
                current_ids.extend(encoding_fonction(model, ", "))
            else:
                current_ids.extend(encoding_fonction(model, "}"))

    params_json = model.decode(current_ids[params_start_idx:])
    return params_json, param_name


def generate_tool_call(
        user_prompt: str,
        model: Small_LLM_Model,
        fonction_allow: list[list[int]],
        tools: dict[str, Any]) -> str:
    """Generate a full ``{"name": ..., "parameters": ...}`` tool call.

    Orchestrates ``select_tool_name`` and ``generate_parameters``
    under constrained decoding, then assembles and cleans the
    resulting JSON.

    Args:
        user_prompt: The natural-language request from the user.
        model: The loaded LLM.
        fonction_allow: Pre-encoded token sequences for every tool.
        tools: Mapping from tool name to its parameter schema.

    Returns:
        A JSON string for the tool call, or ``"error"`` if the
        prompt matches no available tool.
    """
    max_json_number_length = 24
    calls_per_number_param = max_json_number_length
    calls_per_string_param = len(user_prompt) + max_json_number_length

    tool_name, current_ids = select_tool_name(
        user_prompt, model, fonction_allow, tools
    )

    if tool_name == "fn_none" or tool_name not in tools:
        return "error"

    properties = tools[tool_name]
    params_json, param_name = generate_parameters(
        model, current_ids, properties,
        calls_per_number_param, calls_per_string_param
    )

    if properties[param_name]["type"] == "number":
        params = convert_float(params_json)
    else:
        params = params_json
    final_prompt = f'{{"name": "{tool_name}", "parameters": {{{params}}}'

    return clean_json_output(final_prompt)
