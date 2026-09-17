from typing import Any
from src.parsing import Config
from llm_sdk import Small_LLM_Model
from src.generator import generate_tool_call, get_all_tool_encodings
from src.utils import create_json
from src.display import print_color, print_information, print_separator
import json


def process_prompt(
        prompt: str,
        model: Small_LLM_Model,
        fonction_allow: list[list[int]],
        tools: dict[str, Any]) -> dict[str, Any]:
    """Turn a single prompt into a result entry for the output file.

    Args:
        prompt: The natural-language request.
        model: The loaded LLM.
        fonction_allow: Pre-encoded token sequences for every tool.
        tools: Mapping from tool name to its parameter schema.

    Returns:
        A dict with ``prompt`` and either ``name``/``parameters``
        on success, or an ``error`` key describing what went wrong.
    """
    if not prompt or len(prompt.strip()) == 0:
        return {
            "prompt": prompt,
            "error": "Prompt cant be empty",
        }

    tool_call_str = generate_tool_call(prompt, model, fonction_allow, tools)
    if tool_call_str == "error":
        return {
            "prompt": prompt,
            "error": "Prompt not found function",
        }

    try:
        tool_call_dict = json.loads(tool_call_str)
        return {
            "prompt": prompt,
            "name": tool_call_dict["name"],
            "parameters": tool_call_dict["parameters"]
        }
    except json.JSONDecodeError as e:
        return {
            "prompt": prompt,
            "error": str(e),
            "raw_output": tool_call_str
        }


def main() -> None:
    """Run the full function-calling pipeline end to end.

    Parses CLI arguments, loads the model and input files, generates
    a tool call for every prompt with constrained decoding, prints
    each result, and writes them all to the output JSON file.
    """
    config = Config.parse_arguments()
    print_information(
        config.model, config.input, config.output, config.functions_definition
    )

    tools = config.load_tools()

    model = Small_LLM_Model(model_name=config.model)
    fonction_allow = get_all_tool_encodings(model, tools)
    all_prompt = config.load_prompt()

    final_reponse = []
    for prompt in all_prompt:
        data = process_prompt(prompt, model, fonction_allow, tools)
        final_reponse.append(data)
        print_color(data)
    print_separator(200)

    create_json(final_reponse, config.output)


if __name__ == "__main__":

    try:
        main()
    except Exception as e:
        print(e)
