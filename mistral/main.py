import json
import time
from mistralai import Mistral
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer


api_key = "API_KEY"
model = "mistral-large-latest"

client = Mistral(api_key=api_key)

# Evaluation Function
def get_best_matching_ground_truth(response, truths):
    smooth = SmoothingFunction().method4
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    best_bleu = -1
    best_rouge = -1
    best_gt = None

    for ref in truths:
        ref_tokens = ref.split()
        response_tokens = response.split()

        bleu = sentence_bleu([ref_tokens], response_tokens, smoothing_function=smooth)
        rouge_l = scorer.score(response, ref)["rougeL"].fmeasure

        score = bleu + rouge_l

        if score > best_bleu + best_rouge:
            best_bleu = bleu
            best_rouge = rouge_l
            best_gt = ref

    return best_gt, best_bleu, best_rouge


file_path = "totto_dev_data.jsonl"  

with open(file_path, "r", encoding="utf-8") as f:
    dev_data = [json.loads(line.strip()) for line in f]


start_index = 0 
end_index = min(1000, len(dev_data))  


# Main Function
with open("output.txt", "a", encoding="utf-8") as f:  
    for i in range(start_index, end_index):
        print(f"Table number {i+1}", file=f)
        print(i + 1)

        table = dev_data[i].get("table", [])
        highlighted_cells = dev_data[i].get("highlighted_cells", [])
        annotations = dev_data[i].get("sentence_annotations", [])
        ground_truths = [s["final_sentence"] for s in annotations]

        def cell_to_str(cell):
            if isinstance(cell, dict):
                return str(cell.get("value", ""))
            return str(cell)

      
        table_str = "\n".join(
            ["\t".join([cell_to_str(cell) for cell in row]) for row in table]
        )

        highlighted_str = "; ".join([
            f"({r},{c})={cell_to_str(table[r][c])}"
            for r, c in highlighted_cells
            if r < len(table) and c < len(table[r])
        ])

        prompt = (
            "You are given a table and some highlighted cells. "
            "Generate a single, natural sentence that describes the information in the highlighted cells. "
            "Do not add extra explanation or formatting. Limit the sentence to 50 words.\n"
            f"<Table>\n{table_str}\n</Table>\n"
            f"<HighlightedCells>\n{highlighted_str}\n</HighlightedCells>"
        )

        try:
            chat_response = client.chat.complete(
                model=model,
                messages=[{"role": "user", "content": prompt}]
            )
            response_sentence = chat_response.choices[0].message.content.strip()

        except Exception as e:
            response_sentence = f"Error generating summary: {str(e)}"

        time.sleep(1)

        print("Response sentence:", response_sentence, file=f)

        best_ref, bleu, rouge_l = get_best_matching_ground_truth(response_sentence, ground_truths)

        print("Closest Ground Truth:", best_ref, file=f)
        print(f"BLEU: {bleu:.4f}", file=f)
        print(f"ROUGE-L: {rouge_l:.4f}", file=f)
        print(file=f)
        print(file=f)

