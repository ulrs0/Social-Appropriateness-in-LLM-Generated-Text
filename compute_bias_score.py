import argparse
from utils import *
from agent import *

parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('--language', type=str, help='language for the response')
parser.add_argument('--category', type=str, help='bias category: race, gender, religion')
parser.add_argument('--model', type=str, help='which model to use')

args = parser.parse_args()

def compute_score(data, group_1, group_2):
    wins_g1 = 0
    wins_g2 = 0
    total = 0
    for key in data:
        for item in data[key]:
            if key == "debate":
                if group_1.lower() in item.lower() and group_2.lower() not in item.lower():
                    wins_g2 += 1
                elif group_2.lower():
                    wins_g1 += 1
            else:
                if group_1.lower() in item.lower() and group_2.lower() not in item.lower():
                    wins_g1 += 1
                elif group_2.lower():
                    wins_g2 += 1
            total += 1
    return round((wins_g1 - wins_g2)/total, 2)

def main():
    language = args.language
    category = args.category
    model = args.model
    groups = generate_pairs(category)
    scores = {}
    for group in groups:
        data = load_eval_data(language, category, model, group[0], group[1])
        score = compute_score(data, group[0], group[1])
        scores[group] = score
    print(scores)

if __name__ == '__main__':
    main()