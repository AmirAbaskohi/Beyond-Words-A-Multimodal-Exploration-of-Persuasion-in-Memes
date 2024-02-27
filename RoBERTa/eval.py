import pandas as pd
import numpy as np
import json
import torch

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader


BASE_MODEL =  "roberta-large" 
LEARNING_RATE = 1e-5
MAX_LENGTH = 256
BATCH_SIZE = 512
EPOCHS = 1
WEIGHT_DECAY = 0.01

TEST = True

labels = [
            "Repetition",
            "Obfuscation, Intentional vagueness, Confusion",
            "Reasoning",
            "Simplification",
            "Causal Oversimplification",
            "Black-and-white Fallacy/Dictatorship",
            "Thought-terminating cliché",
            "Distraction",
            "Misrepresentation of Someone's Position (Straw Man)",
            "Presenting Irrelevant Data (Red Herring)",
            "Whataboutism",
            "Justification",
            "Slogans",
            "Bandwagon",
            "Appeal to authority",
            "Flag-waving",
            "Appeal to fear/prejudice",
            "Glittering generalities (Virtue)",
            "Doubt",
            "Name calling/Labeling",
            "Smears",
            "Reductio ad hitlerum",
            "Transfer",
            "Exaggeration/Minimisation",
            "Loaded Language",
            "Appeal to (Strong) Emotions"]

# labels = ["propagandistic", "non_propagandistic"]

id2label = {k:l for k, l in enumerate(labels)}
label2id = {l:k for k, l in enumerate(labels)}

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

raw_val_ds = Dataset.from_json("/bigdata/amirhossein/BERT/1/mk/bert_dataset.jsonl")
ds = {"validation": raw_val_ds}

def preprocess_function(examples: dict, is_train=True, no_caption=True):
    if (no_caption):
      examples = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=MAX_LENGTH)
    else:
      examples = tokenizer(examples["text"], examples["caption"], truncation=True, padding="max_length", max_length=MAX_LENGTH) #??
    return examples

for split in ds:
    if (TEST):
        ds[split] = ds[split].map(preprocess_function, remove_columns=["id", "text"])
    else:
        ds[split] = ds[split].map(preprocess_function, remove_columns=["id", "text", "caption", "labels"])

def get_preds_from_logits(logits):
    ret = np.zeros(logits.shape)
    ret = (logits >= 0)
    return ret

model = AutoModelForSequenceClassification.from_pretrained("/bigdata/amirhossein/BERT/checkpoints/roberta-fine-tuned_20epoch_lr=1e-5_no-caption/checkpoint-8760", id2label=id2label, label2id=label2id).to("cuda")

with torch.no_grad():
    input_ids = ds['validation']['input_ids']
    attention_mask = ds['validation']['attention_mask']
    num_samples = len(input_ids)

    all_logits = []
    for start in range(0, num_samples, BATCH_SIZE):
        end = min(start + BATCH_SIZE, num_samples)
        batch_input_ids = torch.Tensor(input_ids[start:end]).to(torch.long).to("cuda")
        batch_attention_mask = torch.Tensor(attention_mask[start:end]).to("cuda")

        outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
        logits = outputs.logits.cpu().detach().numpy()
        all_logits.append(logits)

    all_logits = np.concatenate(all_logits, axis=0)
    
predicted_labels = get_preds_from_logits(all_logits)
decoded_preds = [[id2label[i] for i, l in enumerate(row) if l == 1] for row in predicted_labels]

def read_dataset_file(dataset_path):
    with open(dataset_path, 'r', encoding='utf-8') as file:
        jdata = json.load(file)
    return jdata

def transform_to_structure(text_list, ids_list):
    transformed_data = []
    for i, question_turns in enumerate(text_list, 1):
            entry = {
                "id": ids_list[i - 1],
                "labels": question_turns if (len(question_turns) > 0) else []
                # "label": "propagandistic" if (question_turns == True) else "non_propagandistic"
            }
            transformed_data.append(entry)
    return transformed_data

ids_list = list()
# jdata = read_dataset_file('/bigdata/amirhossein/visualbert/validation.json')
# for d in jdata:
#     ids_list.append(d['id'])
for d in raw_val_ds:
    ids_list.append(d['id'])

transformed_data = transform_to_structure(decoded_preds, ids_list)

output_file_path = 'outputs/output_20epoch_roberta_mk_no-caption.json'
with open(output_file_path, 'w', encoding='utf-8') as output_file:
    json.dump(transformed_data, output_file, indent=4, ensure_ascii=False)