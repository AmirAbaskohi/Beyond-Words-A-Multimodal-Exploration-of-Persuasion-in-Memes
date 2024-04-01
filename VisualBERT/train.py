import torch
import pandas as pd
import numpy as np
import json
from transformers import BertTokenizer, VisualBertModel, VisualBertConfig, VisualBertForVisualReasoning
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, Trainer, TrainerCallback, TrainingArguments
from torch.utils.data import DataLoader
import pickle 
from datasets import Dataset, load_metric



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BASE_MODEL = "bert-base-uncased"
MAX_LENGTH = 512
LEARNING_RATE = 1e-5
BATCH_SIZE = 16
EPOCHS = 20
WEIGHT_DECAY = 0.01


#generate dataset 

labels_list = [
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

id2label = {k:l for k, l in enumerate(labels_list)}
label2id = {l:k for k, l in enumerate(labels_list)}

configuration = VisualBertConfig.from_pretrained('uclanlp/visualbert-vcr-coco-pre', visual_embedding_dim=1024, id2label=id2label, label2id=label2id)
model = VisualBertForVisualReasoning(configuration) #??
model = model.to(device)
tokenizer = BertTokenizer.from_pretrained(BASE_MODEL)

with open('/bigdata/amirhossein/visualbert/visual_embeds_dev.pkl', 'rb') as f:
  visual_embeds_validation = pickle.load(f)

with open('/bigdata/amirhossein/visualbert/visual_embeds_train.pkl', 'rb') as f:
  visual_embeds_train = pickle.load(f)

def preprocess_function(examples, visual_embeds, no_caption=False):
    temp_examples = examples

    labels = [0] * len(id2label)
    for k, l in id2label.items():
        if (examples["labels"] != None and l in examples["labels"]):
            labels[k] = 1
        else:
            labels[k] = 0
    if (no_caption):
      examples = tokenizer.encode_plus(
      examples["text"],
      add_special_tokens=True,
      max_length=MAX_LENGTH,
      return_token_type_ids=False,
      padding="max_length",
      truncation=True,
      return_attention_mask=True,
      return_tensors='pt',
      )
    else:
      examples = tokenizer.encode_plus(
      examples["text"], examples["caption"],
      add_special_tokens=True,
      max_length=MAX_LENGTH,
      return_token_type_ids=False,
      padding="max_length",
      truncation=True,
      return_attention_mask=True,
      return_tensors='pt',
      )

    examples["label"] = labels
    for item in visual_embeds:
      if (temp_examples['file_name'] == item['path'].split('/')[-1]):
        examples['visual_embeds'] = item['embed'].to(device)
        examples['visual_attention_mask'] = torch.ones(examples['visual_embeds'].shape[:-1], dtype=torch.float).to(device)
        break
    else:
        print("not found!!!!!!!!")

    examples["input_ids"] = torch.tensor(examples["input_ids"]).flatten()
    examples["attention_mask"] = torch.tensor(examples["attention_mask"]).flatten()
    return examples

raw_train_ds = Dataset.from_pandas(pd.read_csv('/bigdata/amirhossein/visualbert/visualbert_dataset.csv'))
raw_val_ds = Dataset.from_pandas( pd.read_csv('/bigdata/amirhossein/visualbert/visualbert_dataset_dev.csv'))
ds = {"train": raw_train_ds, "validation": raw_val_ds}
for split in ds:
    ds[split] = ds[split].map(preprocess_function, fn_kwargs={"visual_embeds": visual_embeds_train if (split == 'train') else visual_embeds_validation}, remove_columns=["file_name", "text", "caption", "labels"])

#train

class MyTrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels.float())

        return (loss, outputs) if return_outputs else loss

training_args = TrainingArguments(
    output_dir="./visualbert_checkpoints_new_caption/",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    evaluation_strategy="epoch",
    num_train_epochs=EPOCHS,
    save_strategy="epoch",
    # logging_strategy="steps",
    # load_best_model_at_end=True,
    save_total_limit=20,
    # logging_steps=1,
    weight_decay=WEIGHT_DECAY,
)

trainer = MyTrainer(
    model=model,
    args=training_args,
    train_dataset=ds["train"],
    eval_dataset=ds['validation']
)

trainer.train(resume_from_checkpoint=False)