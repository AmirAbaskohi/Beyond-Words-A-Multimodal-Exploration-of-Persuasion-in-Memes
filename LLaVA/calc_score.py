import json
# from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score
from sacrebleu.metrics import BLEU, CHRF, TER

# Load your model's output and ground truth from the JSONL and JSON files
with open('/bigdata/amirhossein/LLaVA/MemeCap/test/llava_dataset/answer-llm-finetuned-100token.jsonl', 'r') as file:
    model_output_data = [json.loads(line) for line in file]

with open('MemeCap/test/llava_dataset/llava_dataset.json', 'r') as file:
    ground_truth_data = json.load(file)

# Prepare references and hypotheses
references = []
references_bleu = []
hypotheses = [item['text'] for item in model_output_data]

for item in ground_truth_data:
    ground_truth = next((conv['value'] for conv in item['conversations'] if conv['from'] == 'gpt'), '')
    references.append(ground_truth)
    references_bleu.append([ground_truth])

# bleu_score = corpus_bleu(references_bleu, hypotheses)

bleu = BLEU()
bleu_score2 = bleu.corpus_score(hypotheses, references_bleu)


# Calculate ROUGE-L using rouge_score
rouge_scorer_obj = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

# Calculate BERTScore
P, R, F1 = score(hypotheses, references, model_type='microsoft/deberta-xlarge-mnli', lang="en", verbose=False)

# Print results
# print(f"BLEU-4 Score: {bleu_score:.4f}")
print(f"BLEU-4 Score: {bleu_score2.score:.4f}")

s = 0
# s1 = 0
for i in range(len(hypotheses)):
    rouge_scores = rouge_scorer_obj.score(hypotheses[i], references[i])
    s += rouge_scores['rougeL'][2]
    # s1 += sentence_bleu(references[i], hypotheses[i], weights = (0.5, 0.5))#, smoothing_function=smoothie)
    # print((rouge_scores['rougeL'][2]))
# print(s1/ len(hypotheses))
print("ROUGE-L F-Scores avg:", s / len(hypotheses))

# for i in range(len(hypotheses)):
#     print(f"Question {i} F1-BERTScore: {F1[i].item():.4f}")

print(f"F1-BERTScore avg: {F1.mean():.4f}")

