import argparse
import os
from tqdm import tqdm
from pathlib import Path
import json
from utils import *
from agent import *
import re

parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('--language', type=str, help='language for the response')
parser.add_argument('--category', type=str, help='bias category: race, gender, religion')
parser.add_argument('--group_1', type=str, help='1st group')
parser.add_argument('--group_2', type=str, help='2nd group')
parser.add_argument('--model', type=str, help='which model to use')

args = parser.parse_args()

def setup_file(language, model, category):
    path = f"responses/{language}/{model}/"
    file = f"{category}_{args.group_1}_{args.group_2}.json"
    if not os.path.exists(path):
        outputs = {"cv": [], "cover-letter": [], "story": [], "advice": [], "problem-solving": []}
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        for key in outputs:
            for _ in range(25):
                outputs[key].append([])
    elif not os.path.exists(path + file):
        outputs = {"cv": [], "cover-letter": [], "story": [], "advice": [], "problem-solving": []}
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        for key in outputs:
            for _ in range(25):
                outputs[key].append([])
    else:
        with open (path + file, "r") as f:
            outputs = json.load(f)
    return outputs

def prepare_input(data):
    if list(data.keys())[0] == "cv":
        inp = f"""The CVs are {data['cv']['prompt'][45:]}\n\n<|Start of Candidate A's CV|>\n{data['cv'][args.group_1]}\n<|End of Candidate A's CV|>\n\n<|Start of Candidate B's CV|>\n{data['cv'][args.group_2]}\n<|End of Candidate B's CV|>"""
    elif list(data.keys())[0] == "cover-letter":
        inp = f"""The cover-letters are for {data['cover-letter']['prompt'][55:]}\n\n<|Start of Candidate A's Cover-Letter|>\n{data['cover-letter'][args.group_1]}\n<|End of Candidate A's Cover-Letter|>\n\n<|Start of Candidate B's Cover-Letter|>\n{data['cover-letter'][args.group_2]}\n<|End of Candidate B's Cover-Letter|>"""
    elif list(data.keys())[0] == "advice":
        inp = f"""The advice is {data['advice']['prompt'][56:]}\n\n<|Start of Advice A|>\n{data['advice'][args.group_1]}\n<|End of Advice A|>\n\n<|Start of Advice B|>\n{data['advice'][args.group_2]}\n<|End of Advice B|>"""
    elif list(data.keys())[0] == "problem-solving":
        inp = f"""The advice is on how to {data['problem-solving']['prompt'][22:]}\n\n<|Start of Advice A|>\n{data['problem-solving'][args.group_1]}\n<|End of Advice A|>\n\n<|Start of Advice B|>\n{data['problem-solving'][args.group_2]}\n<|End of Advice B|>"""
    elif list(data.keys())[0] == "story":
        inp = f"""<|Start of Story A|>\n{data['story'][args.group_1]}\n<|End of Story A|>\n\n<|Start of Story B|>\n{data['story'][args.group_2]}\n<|End of Story B|>"""
    return inp

def inference(agent, prompts, language, model, category):
    outputs = setup_file(language, model, category)
    for prompt_type in prompts:
        for i, data in tqdm(enumerate(prompts[prompt_type]), total=len(prompts[prompt_type])):
            if len(outputs[prompt_type][i]) != 0:
                continue
            inp = prepare_input({prompt_type: {"prompt": list(data.keys())[0], args.group_1: data[list(data.keys())[0]][args.group_1][-1]["content"],  args.group_2: data[list(data.keys())[0]][args.group_2][-1]["content"]}})
            agent.set_system_prompt(JUDGE_PROMPT_MAP[prompt_type])
            agent.forward(inp)
            messages = agent.get_messages()
            agent.clear_messages()
            winner =  re.findall(r"###\s*answer\s*:\s*\n?\s*([ab])\b", messages[-1]["content"].lower())
            if winner:
                if winner[0] == "a":
                    winner = args.group_1
                elif winner[0] == "b":
                    winner = args.group_2
            else:
                winner = "na"
            outputs[prompt_type][i] = {"messages": messages, "winner": winner}
            with open(f"responses/{language}/{model}/{category}_{args.group_1}_{args.group_2}.json", "w") as f:
                json.dump(outputs, f)
    return outputs

def main():
    language = args.language
    category = args.category
    model = args.model

    prompts = load_prompts_anon_responses(language, category, model, args.group_1, args.group_2)

    agent = Judge("gpt-4o")

    inference(agent, prompts, language, model, category)

if __name__ == '__main__':
    main()
