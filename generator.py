from llm_sdk import Small_LLM_Model
from masking import get_allowed_next_tokens, mask_logits
import numpy as np
from parsing import load_tools
from prompting import build_prompt_find_function
from prompting import format_tools
import re


def clean_json_output(raw_text: str) -> str:
    cleaned = re.sub(r",(\s*,)+", ", ", raw_text)

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

    return cleaned[start:end].strip()

def encoding_fonction(model: Small_LLM_Model, text_to_encode: str):
    new = model.encode(text_to_encode).tolist()[0]
    return new


def get_all_tool_encodings(model, tools):
    all_name_encoding = []
    for to in tools:
        all_name_encoding.append(encoding_fonction(model, to))
    # print(get_allowed_next_tokens(all_name_encoding, [8822, 1889]))
    return all_name_encoding


def all_numbers_tokens(model):
    new = []
    for a in range(10):
        new.append(model.encode(str(a))[0].tolist()[0])
    return new


def all_char_tokens(model):
    new = []
    aplha = "abcdefghijklmnopqrstuvwxz"
    for char in aplha:
        new.append(model.encode(char)[0].tolist()[0])
    return new


def generate_tool_call(
        user_prompt,
        model: Small_LLM_Model,
        fonction_allow,
        tools):
    function_tokens = []
    number_tokens = []
    tools_text = format_tools(tools)

    # print(fonction_allow)
    tensor_ids = model.encode(
        build_prompt_find_function(
            user_prompt, tools_text))
    current_ids = tensor_ids[0].tolist()
    for _ in range(100):
        logits = model.get_logits_from_input_ids(current_ids)
        allowed_tokens = get_allowed_next_tokens(
            fonction_allow, function_tokens)
        if not allowed_tokens:
            break
        final_logit = mask_logits(logits, allowed_tokens)
        token_id = int(np.argmax(final_logit))
        current_ids.append(token_id)
        function_tokens.append(token_id)

    tool_name = model.decode(function_tokens)
    properties = tools[tool_name]
    number_tokens = all_numbers_tokens(model)
    param_keys = list(properties.keys())
    current_ids.extend(encoding_fonction(model, '", "parameters": {'))
    params_start_idx = len(current_ids)
    for idx, param_name in enumerate(param_keys):
        current_ids.extend(encoding_fonction(model, f'"{param_name}": '))
        if properties[param_name]["type"] == "number":
            if idx != len(param_keys) - 1:
                stop_char = ","
            else:
                stop_char = "}"
            stop_token_id = encoding_fonction(model, stop_char)[0]
            allowed_tokens = number_tokens + [stop_token_id]
            for _ in range(20):
                logits = model.get_logits_from_input_ids(current_ids)
                final_logit = mask_logits(logits, allowed_tokens)
                token_id = int(np.argmax(final_logit))
                current_ids.append(token_id)
                if token_id == stop_token_id:
                    break
        if properties[param_name]["type"] == "string":
            current_ids.extend(encoding_fonction(model, '"'))
            stop_token_id = encoding_fonction(model, '"')[0]
            for _ in range(50):
                logits = model.get_logits_from_input_ids(current_ids)
                token_id = int(np.argmax(logits))
                current_ids.append(token_id)
                if '"' in model.decode([token_id]):
                    break
            if idx != len(param_keys) - 1:
                current_ids.extend(encoding_fonction(model, ", "))
            else:
                current_ids.extend(encoding_fonction(model, "}"))
    params_json = model.decode(current_ids[params_start_idx:])
    final_prompt = f'{{"name": "{tool_name}", "parameters": {{{params_json}}}'
    return clean_json_output(final_prompt)
