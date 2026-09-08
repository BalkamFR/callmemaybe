from parsing import load_tools, load_prompt
from llm_sdk import Small_LLM_Model
import json
from generator import generate_tool_call, get_all_tool_encodings



def main():
    model = Small_LLM_Model(model_name="Qwen/Qwen3-0.6B")
    fonction_allow = get_all_tool_encodings(model)
    tools = load_tools()
    tools_text = json.dumps(tools, indent=2)
    all_prompt = load_prompt()
    for prompt in all_prompt:
        print(f"prompt({prompt}), reponse : {generate_tool_call(prompt, model,fonction_allow,tools_text)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)