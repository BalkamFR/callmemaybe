from llm_sdk import Small_LLM_Model
from masking import get_allowed_next_tokens, mask_logits
import numpy as np
from parsing import load_tools
from prompting import build_prompt_find_function
from prompting import format_tools
from pathlib import Path
import re
import json



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
    new.append(model.encode(".")[0].tolist()[0])
    new.append(model.encode("-")[0].tolist()[0])
    for a in range(10):
        new.append(model.encode(str(a))[0].tolist()[0])
    return new


def convert_float(params_json: str) -> str:
    return re.sub(r"(:\s*)(-?\d+)(?![.\d])", r"\1\2.0", params_json)


def is_path_syntax(value: str) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    cleaned_value = value.strip()
    path_obj = Path(cleaned_value)

    if "/" in cleaned_value or "\\" in cleaned_value or path_obj.suffix != "" or path_obj.parent != Path("."):
        return True

    return False

def clean_json_output(raw_text: str) -> str:
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

def generate_tool_call(
        user_prompt,
        model: Small_LLM_Model,
        fonction_allow,
        tools):
    function_tokens = []

    local_allow = list(fonction_allow)
    none_encoding = encoding_fonction(model, "fn_none")
    if none_encoding not in local_allow:
        local_allow.append(none_encoding)

    tools_text = format_tools(tools)
    tensor_ids = model.encode(
        build_prompt_find_function(user_prompt, tools_text)
    )
    current_ids = tensor_ids[0].tolist()

    for _ in range(100):
        logits = model.get_logits_from_input_ids(current_ids)
        allowed_tokens = get_allowed_next_tokens(local_allow, function_tokens)
        if not allowed_tokens:
            break
        final_logit = mask_logits(logits, allowed_tokens)
        token_id = int(np.argmax(final_logit))
        current_ids.append(token_id)
        function_tokens.append(token_id)

    tool_name = model.decode(function_tokens)

    if tool_name == "fn_none" or tool_name not in tools:
        return "error"

    properties = tools[tool_name]
    number_tokens = all_numbers_tokens(model)
    param_keys = list(properties.keys())
    current_ids.extend(encoding_fonction(model, '", "parameters": {'))
    params_start_idx = len(current_ids)
    for idx, param_name in enumerate(param_keys):
        current_ids.extend(encoding_fonction(model, f'"{param_name}": '))
        if properties[param_name]["type"] in ("number", "integer"):
            stop_char = "," if idx != len(param_keys) - 1 else "}"
            stop_token_id = encoding_fonction(model, stop_char)[0]
            allowed_tokens = number_tokens + [stop_token_id]
            for _ in range(20):
                logits = model.get_logits_from_input_ids(current_ids)
                final_logit = mask_logits(logits, allowed_tokens)
                token_id = int(np.argmax(final_logit))
                current_ids.append(token_id)
                if token_id == stop_token_id:
                    break
        elif properties[param_name]["type"] == "string":
            current_ids.extend(encoding_fonction(model, '"'))
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
    if properties[param_name]["type"] == "number":
        final_prompt = f'{{"name": "{tool_name}", "parameters": {{{convert_float(params_json)}}}'
    else:
        final_prompt = f'{{"name": "{tool_name}", "parameters": {{{params_json}}}'
    
    return clean_json_output(final_prompt)