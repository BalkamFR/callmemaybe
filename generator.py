from llm_sdk import Small_LLM_Model
from masking import get_allowed_next_tokens, mask_logits
import numpy as np
from parsing import load_tools

def encoding_fonction(model:Small_LLM_Model, text_to_encode:str):
    new = model.encode(text_to_encode).tolist()[0]
    return new


def get_all_tool_encodings(model):
    tools = load_tools()
    all_name_encoding = []
    for to in tools:
        all_name_encoding.append(encoding_fonction(model, to))
    # print(get_allowed_next_tokens(all_name_encoding, [8822, 1889]))
    return all_name_encoding

def generate_tool_call(user_prompt, model, fonction_allow, tools_text):
    function_tokens = []
    phrase = []

    formatted_prompt = (
            "You are a helpful assistant with access to the following tools.\n"
            "Choose the appropriate tool to answer the user request.\n\n"
            f"Available tools:\n{tools_text}\n\n"
            f"User: {user_prompt}\n"
            'Assistant: {"name": "'
        )
    # print(fonction_allow)
    tensor_ids = model.encode(formatted_prompt)
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
        phrase.append(token_id)
    answer = model.decode(phrase)
    return answer

