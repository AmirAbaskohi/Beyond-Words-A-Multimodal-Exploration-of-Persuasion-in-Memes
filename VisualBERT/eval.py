import torch
import pandas as pd
import numpy as np
import json
from transformers import BertTokenizer, VisualBertModel, VisualBertConfig, VisualBertForVisualReasoning, VisualBertForMultipleChoice
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, Trainer, TrainerCallback, TrainingArguments
from torch.utils.data import DataLoader
import pickle 
from datasets import Dataset, load_metric


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BASE_MODEL = "bert-base-uncased"
MAX_LENGTH = 512


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

configuration = VisualBertConfig.from_pretrained('/bigdata/amirhossein/visualbert/visualbert_checkpoints_new_caption/checkpoint-8760', visual_embedding_dim=1024, id2label=id2label, label2id=label2id)
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

raw_val_ds = Dataset.from_pandas( pd.read_csv('/bigdata/amirhossein/visualbert/visualbert_dataset_dev.csv'))
ds = {"validation": raw_val_ds}
for split in ds:
    ds[split] = ds[split].map(preprocess_function, fn_kwargs={"visual_embeds": visual_embeds_train if (split == 'train') else visual_embeds_validation}, remove_columns=["file_name", "text", "caption", "labels"])


 #eval
def get_preds_from_logits(logits):
    ret = np.zeros(logits.shape)
    ret = (logits >= 0)
    return ret
    
batch_size = 16

input_ids_list = ds['validation']['input_ids']
attention_mask_list = ds['validation']['attention_mask']
visual_embeds_list = ds['validation']['visual_embeds']
visual_attention_mask_list = ds['validation']['visual_attention_mask']

num_samples = len(input_ids_list)
num_batches = (num_samples + batch_size - 1) // batch_size  # Calculate the number of batches

model.eval()

all_predicted_labels = []

with torch.no_grad():
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_samples)

        input_ids = torch.tensor(input_ids_list[start_idx:end_idx]).to(device)
        attention_mask = torch.tensor(attention_mask_list[start_idx:end_idx]).to(device)
        visual_embeds = torch.tensor(visual_embeds_list[start_idx:end_idx]).to(device)
        visual_attention_mask = torch.tensor(visual_attention_mask_list[start_idx:end_idx]).to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            visual_embeds=visual_embeds,
            visual_attention_mask=visual_attention_mask,
        )

        logits = outputs.logits.cpu().detach().numpy()
        predicted_labels = get_preds_from_logits(logits)
        all_predicted_labels.extend(predicted_labels)

# Now you have predictions for the entire dataset
decoded_preds = [
    [id2label[i] for i, l in enumerate(row) if l == 1] for row in all_predicted_labels
]

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
jdata = read_dataset_file('/bigdata/amirhossein/visualbert/dev_subtask2a_en.json')
for d in jdata:
    ids_list.append(d['id'])

transformed_data = transform_to_structure(decoded_preds, ids_list)

output_file_path = 'outputs/output_20epoch_new_caption2.json'
with open(output_file_path, 'w', encoding='utf-8') as output_file:
    json.dump(transformed_data, output_file, indent=4, ensure_ascii=False)