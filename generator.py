from llm_sdk import Small_LLM_Model
from masking import get_allowed_next_tokens, mask_logits
import numpy as np
from parsing import load_tools
from prompting import build_prompt_find_function
from prompting import format_tools


def encoding_fonction(model:Small_LLM_Model, text_to_encode:str):
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

def generate_tool_call(user_prompt, model:Small_LLM_Model, fonction_allow, tools):
    function_tokens = []
    number_tokens = []
    tools_text = format_tools(tools)

    # print(fonction_allow)
    tensor_ids = model.encode(build_prompt_find_function(user_prompt, tools_text))
    current_ids = tensor_ids[0].tolist()
    for _ in range(100):
        logits = model.get_logits_from_input_ids(current_ids)
        allowed_tokens = get_allowed_next_tokens(fonction_allow, function_tokens)
        if not allowed_tokens:
            break
        final_logit = mask_logits(logits, allowed_tokens)
        token_id = int(np.argmax(final_logit))
        current_ids.append(token_id)
        function_tokens.append(token_id)


    tool_name = model.decode(function_tokens)
    properties = tools[tool_name]
    param_tokens = []
    number_tokens = all_numbers_tokens(model)
    char_tokens = all_char_tokens(model)
    param_keys = list(properties.keys())
    current_ids.extend(encoding_fonction(model, '", "parameters": {'))
    for idx, param_name in enumerate(param_keys):
        current_ids.extend(encoding_fonction(model, f'"{param_name}": '))
        if idx != len(param_keys) - 1:
            stop_char = ","
        else:
            stop_char = "}"

        stop_token_id = encoding_fonction(model, stop_char)[0]
        if properties[param_name]["type"] == "number":
            allowed_tokens = number_tokens + [stop_token_id]
            for _ in range(20):
                logits = model.get_logits_from_input_ids(current_ids)
                final_logit = mask_logits(logits, allowed_tokens)
                token_id = int(np.argmax(final_logit))
                current_ids.append(token_id)
                param_tokens.append(token_id)
                if token_id == stop_token_id:
                    break
        if properties[param_name]["type"] == "string":
            allowed_tokens = char_tokens + [stop_token_id]
            for _ in range(20):
                logits = model.get_logits_from_input_ids(current_ids)
                final_logit = mask_logits(logits, allowed_tokens)
                token_id = int(np.argmax(final_logit))
                current_ids.append(token_id)
                param_tokens.append(token_id)
                if token_id == stop_token_id:
                    break
    current_ids.extend(encoding_fonction(model, "}"))
    final_prompt = f'{{"name": "{tool_name}", "parameters": {model.decode(param_tokens)}}}'
    return final_prompt
