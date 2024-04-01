import json
from rouge_score import rouge_scorer
from bert_score import score
from sacrebleu.metrics import BLEU

# Load your model's output and ground truth from the JSONL and JSON files
with open('answers.jsonl', 'r') as file:
    model_output_data = [json.loads(line) for line in file]

with open('memes-test.json', 'r') as file:
    ground_truth_data = json.load(file)

references = []
references_bleu = []
hypotheses = [item['text'] for item in model_output_data]

for item in ground_truth_data:
    ground_truth = next((conv['value'] for conv in item['conversations'] if conv['from'] == 'gpt'), '')
    references.append(ground_truth)
    references_bleu.append([ground_truth])

bleu = BLEU()
bleu_score2 = bleu.corpus_score(hypotheses, references_bleu)


rouge_scorer_obj = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

s = 0
for i in range(len(hypotheses)):
    rouge_scores = rouge_scorer_obj.score(hypotheses[i], references[i])
    s += rouge_scores['rougeL'][2]
print("ROUGE-L F-Scores avg:", s / len(hypotheses))

P, R, F1 = score(hypotheses, references, model_type='microsoft/deberta-xlarge-mnli', lang="en", verbose=False)

print(f"BLEU-4 Score: {bleu_score2.score:.4f}")

print(f"F1-BERTScore avg: {F1.mean():.4f}")