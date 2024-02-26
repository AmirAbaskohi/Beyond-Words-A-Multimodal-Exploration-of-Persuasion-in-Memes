import torch
import pandas as pd
import numpy as np
from transformers import BertTokenizer, VisualBertModel, VisualBertConfig
import pickle 
from datasets import Dataset, load_metric

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BASE_MODEL = "bert-base-uncased"
MAX_LENGTH = 256

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
tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")

with open('/bigdata/amirhossein/visualbert/visual_embeds_validation.pkl', 'rb') as f:
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
        print("not found")

    examples["input_ids"] = torch.tensor(examples["input_ids"]).flatten()
    examples["attention_mask"] = torch.tensor(examples["attention_mask"]).flatten()
    return examples

raw_train_ds = Dataset.from_pandas(pd.read_csv('/bigdata/amirhossein/visualbert/visualbert_dataset_train.csv'))
raw_val_ds = Dataset.from_pandas( pd.read_csv('/bigdata/amirhossein/visualbert/visualbert_dataset_valid.csv'))
ds = {"train": raw_train_ds, "validation": raw_val_ds}
for split in ds:
    ds[split] = ds[split].map(preprocess_function, fn_kwargs={"visual_embeds": visual_embeds_train if (split == 'train') else visual_embeds_validation}, remove_columns=["file_name", "text", "caption", "labels"])

with open('/bigdata/amirhossein/visualbert/dataset.pkl', 'wb') as f:
    pickle.dump(ds, f)