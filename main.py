from parsing import load_tools, load_prompt
from llm_sdk import Small_LLM_Model
import json
from generator import generate_tool_call, get_all_tool_encodings
import time

def main():
    model = Small_LLM_Model(model_name="Qwen/Qwen3-0.6B")
    tools = load_tools()
    fonction_allow = get_all_tool_encodings(model, tools)
    all_prompt = load_prompt()
    for prompt in all_prompt:
        reponse = f"prompt({prompt}), reponse : {generate_tool_call(prompt, model,fonction_allow,tools)}"
        for char in str(reponse):
            print(char, end="", flush=True,)
            time.sleep(0.01)
        print("\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)