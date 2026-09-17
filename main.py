from parsing import load_tools, load_prompt, parse_arguments
from llm_sdk import Small_LLM_Model
from generator import generate_tool_call, get_all_tool_encodings
from create_json import create_json
import json


def main():
    config = parse_arguments()

    model = Small_LLM_Model(model_name="Qwen/Qwen3-0.6B")
    tools = load_tools(config["functions_definition"])
    
    fonction_allow = get_all_tool_encodings(model, tools)
    all_prompt = load_prompt(config["input"])
    final_reponse = []
    for prompt in all_prompt:
        if not prompt or len(prompt.strip()) == 0:
            data = {
                "prompt": prompt,
                "error": "Prompt cant be empty",
            }
            final_reponse.append(data)
            continue
        tool_call_str = generate_tool_call(prompt, model, fonction_allow, tools)
        if tool_call_str == "error":
            data = {
                "prompt": prompt,
                "error": "Prompt not fund function",
            }
            final_reponse.append(data)
            print(data)

            continue
        try:
            tool_call_dict = json.loads(tool_call_str)
            data = {
                "prompt": prompt,
                "name": tool_call_dict["name"],
                "parameters": tool_call_dict["parameters"]
            }
        except json.JSONDecodeError as e:
            data = {
                "prompt": prompt,
                "error": str(e),
                "raw_output": tool_call_str
            }
        final_reponse.append(data)
        print(data)
    create_json(final_reponse, config["output"])

if __name__ == "__main__":

    try:
        main()
    except Exception as e:
        print(e)
