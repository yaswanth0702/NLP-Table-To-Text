import json
import time

import requests
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer


def get_best_matching_ground_truth(response, truths):
    smooth = SmoothingFunction().method4
    rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    best_bleu = -1
    best_rouge = -1
    best_gt = None

    for ref in truths:
        ref_tokens = ref.split()
        response_tokens = response.split()

        bleu = sentence_bleu([ref_tokens], response_tokens, smoothing_function=smooth)
        rouge_l = rouge.score(ref, response)["rougeL"].fmeasure

        score = bleu + rouge_l

        if score > best_bleu + best_rouge:
            best_bleu = bleu
            best_rouge = rouge_l
            best_gt = ref

    return best_gt, best_bleu, best_rouge

file_path = "../totto_dev_data.jsonl"

smooth = SmoothingFunction().method1

with open(file_path, "r", encoding="utf-8") as f:
    dev_data = [json.loads(line) for line in f]

with open("gemini_output.txt", "w", encoding="utf-8") as f:

    for i in range(1000):
        print(f"Table number {i+1}", file=f)
        print(i+1)
        table_prompt = dev_data[i]["table"]
        highlighted_cells_prompt = dev_data[i]["highlighted_cells"]
        sentence_annotations = dev_data[i]['sentence_annotations']
        ground_truths = []
        for sentence in sentence_annotations:
            ground_truths.append(sentence["final_sentence"])

        print("Highlighted Cells Prompt", highlighted_cells_prompt, file=f)
        print("Table Prompt: ", table_prompt, file=f)

        api_key = "api_key"
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"

        headers = {
            "Content-Type": "application/json"
        }

        data = {
            "system_instruction": {
                "parts": [
                    {"text": "You are given a table and some highlighted cells. Generate a natural sentence describing the information in the highlighted cells."}
                ]
            },
            "contents": [
                {
                    "parts": [
                        {"text": f"<Table>{table_prompt}</Table> <Highlighted Cells>{highlighted_cells_prompt}</HighlightedCells>"}
                    ]
                }
            ]
        }

        response = requests.post(url, headers=headers, json=data)

        response_sentence = response.json()['candidates'][0]['content']['parts'][0]['text']
        print("Response sentence:", response_sentence, file=f)


        best_ref, bleu, rouge = get_best_matching_ground_truth(response_sentence, ground_truths)
        print("Closest Ground Truth:", best_ref, file=f)
        print("BLEU:", bleu, file=f)
        print("ROUGE-L:", rouge, file=f)

        print(file=f)
        print(file=f)
        time.sleep(5)

