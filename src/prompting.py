from typing import Any
import json


def format_tools(tools: dict[str, Any]) -> str:
    """Serialize the available tools as a pretty-printed JSON string."""
    return json.dumps(tools, indent=2)


def build_prompt_find_function(user_prompt: str, tools_text: str) -> str:
    """Build the few-shot prompt used to pick a tool name for a request.

    Args:
        user_prompt: The natural-language request from the user.
        tools_text: JSON-formatted description of the available tools.

    Returns:
        The full prompt, ending right after the opening
        ``{"name": "`` so the model only has to complete the name.
    """
    fallback_tool_desc = (
        "- fn_none: Use this function when no other tool matches."
    )

    return (
        "You are an expert assistant with access to tools. "
        "Select the correct tool and extract parameters "
        "exactly from the user.\n"
        "Rules:\n"
        "1. If no tool matches, select 'fn_none'.\n"
        "2. Extract values verbatim from the prompt.\n"
        "3. Preserve negative numbers strictly (e.g., -4 must remain -4).\n"
        "4. In strings, copy text literally and escape internal double quotes "
        'with a backslash (\\").\n\n'
        f"Available tools:\n{tools_text}\n{fallback_tool_desc}\n\n"
        "Examples:\n"
        "User: Is -8 an even number?\n"
        'Assistant: {"name": "fn_is_even", "parameters": {"n": -8}}\n\n'
        'User: Format template: Quote: "test" in {var}\n'
        'Assistant: {"name": "fn_format_template", "parameters": '
        '{"template": "Quote: \\"test\\" in {var}"}}\n\n'
        "User: hello\n"
        'Assistant: {"name": "fn_none"}\n\n'
        f"User: {user_prompt}\n"
        'Assistant: {"name": "'
    )
