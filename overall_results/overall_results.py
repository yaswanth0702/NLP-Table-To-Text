import re

with open('overall_results_summary.txt', 'w', encoding='utf-8') as out:

    with open('../bart/bart_output.txt', 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    # Extract BLEU values
    bleu_scores = re.findall(r'BLEU:\s*([0-9.]+)', content)
    bleu_floats = [float(score) for score in bleu_scores]
    total_bleu = sum(bleu_floats)

    # Extract ROUGE-L values
    rouge_scores = re.findall(r'ROUGE-L:\s*([0-9.]+)', content)
    rouge_floats = [float(score) for score in rouge_scores]
    total_rouge = sum(rouge_floats)

    # Print results
    print("Bart Results", file=out)
    print(f"Average Bleu score: {sum(bleu_floats)/len(bleu_floats):.4f}", file=out)
    print(f"Average Rouge score: {sum(rouge_floats)/len(rouge_floats):.4f}", file=out)
    print(file=out)

    with open('../gemini/gemini_output.txt', 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    # Extract BLEU values
    bleu_scores = re.findall(r'BLEU:\s*([0-9.]+)', content)
    bleu_floats = [float(score) for score in bleu_scores]
    total_bleu = sum(bleu_floats)

    # Extract ROUGE-L values
    rouge_scores = re.findall(r'ROUGE-L:\s*([0-9.]+)', content)
    rouge_floats = [float(score) for score in rouge_scores]
    total_rouge = sum(rouge_floats)

    # Print results
    print("Gemini Results", file=out)
    print(f"Average Bleu score: {sum(bleu_floats)/len(bleu_floats):.4f}", file=out)
    print(f"Average Rouge score: {sum(rouge_floats)/len(rouge_floats):.4f}", file=out)
    print(file=out)


    with open('../mistral/output.txt', 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    # Extract BLEU values
    bleu_scores = re.findall(r'BLEU:\s*([0-9.]+)', content)
    bleu_floats = [float(score) for score in bleu_scores]
    total_bleu = sum(bleu_floats)

    # Extract ROUGE-L values
    rouge_scores = re.findall(r'ROUGE-L:\s*([0-9.]+)', content)
    rouge_floats = [float(score) for score in rouge_scores]
    total_rouge = sum(rouge_floats)

    # Print results
    print("Mistral Results", file=out)
    print(f"Average Bleu score: {sum(bleu_floats)/len(bleu_floats):.4f}", file=out)
    print(f"Average Rouge score: {sum(rouge_floats)/len(rouge_floats):.4f}", file=out)
