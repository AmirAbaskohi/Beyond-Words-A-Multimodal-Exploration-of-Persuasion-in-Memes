import pandas as pd
import numpy as np
import json
import torch

from datasets import Dataset, load_metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, Trainer, TrainerCallback, TrainingArguments
from torch.utils.data import DataLoader



BASE_MODEL = "roberta-large"
LEARNING_RATE = 1e-5
MAX_LENGTH = 256
BATCH_SIZE = 16
EPOCHS = 20
WEIGHT_DECAY = 0.01

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

id2label = {k:l for k, l in enumerate(labels)}
label2id = {l:k for k, l in enumerate(labels)}
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, id2label=id2label, label2id=label2id)

raw_train_ds = Dataset.from_json("/arc/project/st-wangll05-1/amir/BERT/bert_dataset_new-caption.jsonl")
raw_val_ds = Dataset.from_json("/arc/project/st-wangll05-1/amir/BERT/bert_dataset_test.jsonl")

ds = {"train": raw_train_ds, "validation": raw_val_ds}

def preprocess_function(examples: dict, no_caption=True):
    labels = [0] * len(id2label)
    for k, l in id2label.items():
        if (l in examples["labels"]):
            labels[k] = 1
        else:
            labels[k] = 0
    if (no_caption):
      examples = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=MAX_LENGTH)
    else:
        examples = tokenizer(examples["text"], examples["caption"], truncation=True, padding="max_length", max_length=MAX_LENGTH)
            
    examples["label"] = labels
    return examples

for split in ds:
    ds[split] = ds[split].map(preprocess_function, remove_columns=["id", "text", "caption", "labels"])


class MyTrainerCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, logs=None, **kwargs):
        print(f"Epoch {state.epoch} ")


class MyTrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.callback = MyTrainerCallback()

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels.float())

        return (loss, outputs) if return_outputs else loss
    
training_args = TrainingArguments(
    output_dir=f"/scratch/st-wangll05-1/amir/BERT/checkpoints/roberta-fine-tuned_{EPOCHS}epoch_lr=1e-5_no-caption_{MAX_LENGTH}/",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    # save_total_limit=1,
    # logging_steps=1,
    weight_decay=WEIGHT_DECAY,
)

trainer = MyTrainer(
    model=model,
    args=training_args,
    train_dataset=ds["train"],
    eval_dataset=ds["validation"],
)

trainer.train(resume_from_checkpoint = False)

def get_preds_from_logits(logits):
    ret = np.zeros(logits.shape)
    ret = (logits >= 0)
    return ret

model.eval()
with torch.no_grad():
    input_ids = ds['validation']['input_ids']
    attention_mask = ds['validation']['attention_mask']

    outputs = model(input_ids=torch.Tensor(input_ids).to(torch.long).to("cuda"), attention_mask=torch.Tensor(attention_mask).to("cuda"))
    logits = outputs.logits.cpu().detach().numpy()
    predicted_labels = get_preds_from_logits(logits)
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
            }
            transformed_data.append(entry)
    return transformed_data

ids_list = list()
jdata = read_dataset_file('/arc/project/st-wangll05-1/amir/BERT/validation.json')
for d in jdata:
    ids_list.append(d['id'])

transformed_data = transform_to_structure(decoded_preds, ids_list)

output_file_path = f'/scratch/st-wangll05-1/amir/BERT/outputs/output_{EPOCHS}epoch_roberta_no_caption_{MAX_LENGTH}.json'
with open(output_file_path, 'w', encoding='utf-8') as output_file:
    json.dump(transformed_data, output_file, indent=4, ensure_ascii=False)