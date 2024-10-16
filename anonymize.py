import argparse
import os
from tqdm import tqdm
from pathlib import Path
import json
from utils import *
from agent import *

parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('--language', type=str, help='language for the response')
parser.add_argument('--category', type=str, help='bias category: race, gender, religion')
parser.add_argument('--model', type=str, help='which model to use')

args = parser.parse_args()

def setup_file(language, model, category, prompts):
    path = f"responses/{language}/{model}/"
    file = f"{category}_anon.json"
    if not os.path.exists(path):
        outputs = []
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        for _ in range(len(prompts)):
            outputs.append([])
    elif not os.path.exists(path + file):
        outputs = []
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        for _ in range(len(prompts)):
            outputs.append([])
    else:
        with open (path + file, "r") as f:
            outputs = json.load(f)
    return outputs

def inference(agent, prompts, language, model, category):
    outputs = setup_file(language, model, category, prompts)
    for i, data in tqdm(enumerate(prompts), total=len(prompts)):
        if len(outputs[i]) != 0:
            continue
        else:
            print("HERE")
        if data[1] != "competition" and data[1] != "debate":
            agent.set_system_prompt(PROMPT_MAP[data[1]])
            agent.forward(data[-1][-1]["content"])
            messages = data[-1]
            messages.extend(agent.get_messages()[2:])
        else:
            messages = data[-1]
        agent.clear_messages()
        outputs[i] = messages
        with open(f"responses/{language}/{model}/{category}_anon.json", "w") as f:
            json.dump(outputs, f)

def main():
    language = args.language
    category = args.category
    model = args.model

    prompts = load_prompts_responses(language, category, model)

    agent = Anonymizer("gpt-4o-mini")

    inference(agent, prompts, language + "1", model, category)

if __name__ == '__main__':
    main()
