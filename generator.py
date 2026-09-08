from llm_sdk import Small_LLM_Model
from masking import get_allowed_next_tokens, mask_logits
import numpy as np
from parsing import load_tools
from prompting import build_prompt
from prompting import format_tools


def encoding_fonction(model:Small_LLM_Model, text_to_encode:str):
    new = model.encode(text_to_encode).tolist()[0]
    return new


def get_all_tool_encodings(model, tools=load_tools()):
    all_name_encoding = []
    for to in tools:
        all_name_encoding.append(encoding_fonction(model, to))
    # print(get_allowed_next_tokens(all_name_encoding, [8822, 1889]))
    return all_name_encoding

def generate_tool_call(user_prompt, model, fonction_allow, tools):
    function_tokens = []
    answer = ""
    tools_text = format_tools(tools)

    # print(fonction_allow)
    tensor_ids = model.encode(build_prompt(user_prompt, tools_text))
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
    final_prompt = f'{{"name": "{tool_name}", "parameters": {properties}}}'
    return final_prompt

