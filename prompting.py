import json


def format_tools(tools: dict) -> str:
    return json.dumps(tools, indent=2)



def build_prompt_find_function(user_prompt: str, tools_text: str) -> str:
    return (
        "You are a helpful assistant with access to the following tools.\n"
        "Choose the appropriate tool to answer the user request.\n\n"
        f"Available tools:\n{tools_text}\n\n"
        f"User: {user_prompt}\n"
        'Assistant: {"name": "'
    )

