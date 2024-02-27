import json
import numpy as np
import pandas as pd
import os
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import functools
from collections import Counter
from argparse import Namespace
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import pandas
import clip
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from datasets import Dataset
from torch.utils.data import DataLoader
# from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
import datasets

datasets.config.HF_DATASETS_OFFLINE = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BASE_MODEL = "roberta-large" 
MAX_LENGTH = 512
LEARNING_RATE = 1e-5
BATCH_SIZE = 8
EPOCHS = 20
WEIGHT_DECAY = 0.01

args = Namespace()
args.savedir = os.path.join('model_save_clip', 'concat_bert_new_caption2')
os.makedirs(args.savedir, exist_ok=True)
args.data_path =  "/arc/project/st-wangll05-1/amir/concat_bert"

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


def preprocess_function(examples, no_caption=False):
    labels = [0] * len(id2label)
    for k, l in id2label.items():
        if (examples["labels"] != None and l in examples["labels"]):
            labels[k] = 1
        else:
            labels[k] = 0

    examples["label"] = labels
    examples["img"] = examples['file_name']
    return examples

raw_train_ds = Dataset.from_pandas(pd.read_csv('/arc/project/st-wangll05-1/amir/concat_bert/visualbert_dataset.csv'))
raw_val_ds = Dataset.from_pandas(pd.read_csv('/arc/project/st-wangll05-1/amir/concat_bert/visualbert_dataset_dev.csv'))
ds = {"train": raw_train_ds, "validation": raw_val_ds}
for split in ds:
    ds[split] = ds[split].map(preprocess_function, remove_columns=["file_name", "labels"])

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
args.n_classes = len(labels_list) 
clip_model, model_transforms = clip.load("ViT-L/14@336px", device=device)


from torch.utils.data import Dataset
class JsonlDataset(Dataset):
    def __init__(self, data, data_path, tokenizer, transforms, args):
        self.data = data
        self.data_dir = data_path 
        self.tokenizer = tokenizer
        self.args = args
        self.n_classes = args.n_classes 

        self.max_seq_len = args.max_seq_len
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        tokenized = self.tokenizer(
            self.data[index]["text"], self.data[index]["caption"],
            add_special_tokens=True,
            max_length=MAX_LENGTH,
            return_token_type_ids=True,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
      ) 
        sentence = tokenized['input_ids'][0]
        mask = tokenized['attention_mask']
        segment = tokenized['token_type_ids'] 

        label = torch.tensor(self.data[index]["label"])

        if self.data[index]["img"]:
            image = Image.open(
                os.path.join(self.data_dir, self.data[index]["img"])
            )

        image = self.transforms(image).unsqueeze(0).to(device)
        with torch.no_grad():
            image = clip_model.encode_image(image)

        return sentence, mask, segment, image, label
    
def collate_fn(batch, args):
    lens = [len(row[0]) for row in batch]
    bsz, max_seq_len = len(batch), max(lens)
    mask_tensor = torch.zeros(bsz, max_seq_len).long()
    text_tensor = torch.zeros(bsz, max_seq_len).long()
    segment_tensor = torch.zeros(bsz, max_seq_len).long()

    img_tensor = torch.stack([row[3] for row in batch])

    tgt_tensor = torch.stack([row[4] for row in batch])

    for i_batch, (input_row, length) in enumerate(zip(batch, lens)):
        tokens, masks, segment = input_row[:3]
        text_tensor[i_batch, :length] = tokens
        segment_tensor[i_batch, :length] = segment
        mask_tensor[i_batch, :length] = masks

    return text_tensor, segment_tensor, mask_tensor, img_tensor, tgt_tensor

args.max_seq_len = MAX_LENGTH 
args.batch_sz = BATCH_SIZE

train = JsonlDataset(ds['train'],
    os.path.join(args.data_path, "train_images"),
    tokenizer,
    model_transforms,
    args,
)

args.train_data_len = len(train)

dev = JsonlDataset(ds['validation'],
    '/arc/project/st-wangll05-1/amir/concat_bert/dev_images',
    tokenizer,
    model_transforms,
    args,
)

collate = functools.partial(collate_fn, args=args)

train_loader = DataLoader(
    train,
    batch_size=args.batch_sz,
    shuffle=True,
    collate_fn=collate,
    drop_last=True,
)

val_loader = DataLoader(
    dev,
    batch_size=args.batch_sz,
    shuffle=False,
    collate_fn=collate,
)

class RoBERTaEncoder(nn.Module):
    def __init__(self, args):
        super(RoBERTaEncoder, self).__init__()
        self.args = args
        self.roberta = AutoModel.from_pretrained(BASE_MODEL)

    def forward(self, txt, mask, segment):
        out = self.roberta(
            txt,
            token_type_ids=segment,
            attention_mask=mask,
            output_hidden_states=False,
        )
        return out.pooler_output

class MultimodalConcatRoBERTaClf(nn.Module):
    def __init__(self, args):
        super(MultimodalConcatRoBERTaClf, self).__init__()
        self.args = args
        self.txtenc = RoBERTaEncoder(args)
        # self.imgenc = ImageEncoder(args)

        last_size = args.hidden_sz + (args.img_hidden_sz * args.num_image_embeds)
        self.clf = nn.ModuleList()

        self.clf.append(nn.Linear(last_size, args.n_classes))

    def forward(self, txt, mask, segment, img):
        txt = self.txtenc(txt, mask, segment)
        # img = self.imgenc(img)
        img = torch.flatten(img, start_dim=1)
        out = torch.cat([txt, img], -1)
        for layer in self.clf:
            out = layer(out)
        return out
    
args.img_embed_pool_type = "avg"
args.num_image_embeds = 1
args.hidden_sz = 1024
args.img_hidden_sz = 768

model = MultimodalConcatRoBERTaClf(args)

criterion = None #nn.BCEWithLogitsLoss(pos_weight=label_weights.cuda())

args.lr = LEARNING_RATE
optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)

args.lr_patience = 2
args.lr_factor = 0.5
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "max", patience=args.lr_patience, verbose=True, factor=args.lr_factor
    )

model.cuda()
torch.save(args, os.path.join(args.savedir, "args.pt"))
start_epoch, global_step, n_no_improve, best_metric = 0, 0, 0, -np.inf

if os.path.exists(os.path.join(args.savedir, "checkpoint.pt")):
        checkpoint = torch.load(os.path.join(args.savedir, "checkpoint.pt"))
        start_epoch = checkpoint["epoch"]
        n_no_improve = checkpoint["n_no_improve"]
        best_metric = checkpoint["best_metric"]
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])

args.max_epochs = EPOCHS
args.gradient_accumulation_steps = 1
args.patience = 20


def get_preds_from_logits(logits):
    ret = np.zeros(logits.shape)
    ret = (logits >= 0)
    return ret

def model_eval(i_epoch, data, model, args, criterion):
    with torch.no_grad():
        losses, preds, tgts = [], [], []
        for batch in data:
            txt, segment, mask, img, tgt = batch

            txt, img = txt.cuda(), img.cuda()
            mask, segment = mask.cuda(), segment.cuda()
            tgt = tgt.cuda()
            out = model(txt, mask, segment, img)
            logits = out
            loss = torch.nn.functional.binary_cross_entropy_with_logits(out, tgt.float()) #criterion(out, tgt)
            losses.append(loss.item())

            pred = get_preds_from_logits(logits)
            preds.append(pred.tolist())


    metrics = {"loss": np.mean(losses)}
    preds = np.vstack(preds)
    return preds, metrics

import shutil

def save_checkpoint(state, is_best, checkpoint_path, filename="checkpoint.pt"):
    filename = os.path.join(checkpoint_path, str(state['epoch']) + filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(checkpoint_path, "model_best.pt"))

def load_checkpoint(model, path):
    best_checkpoint = torch.load(path)
    model.load_state_dict(best_checkpoint["state_dict"])

    
for i_epoch in range(start_epoch, args.max_epochs):
    train_losses = []
    model.train()
    optimizer.zero_grad()

    for batch in tqdm(train_loader, total=len(train_loader)):
        txt, segment, mask, img, tgt = batch

        txt, img = txt.cuda(), img.cuda()
        mask, segment = mask.cuda(), segment.cuda()
        tgt = tgt.cuda()
        out = model(txt, mask, segment, img)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(out, tgt.float()) #criterion(out, tgt)

        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        train_losses.append(loss.item())
        loss.backward()
        global_step += 1
        if global_step % args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

    model.eval()
    preds, metrics = model_eval(i_epoch, val_loader, model, args, criterion)
    print("Train Loss: {:.4f} | Valid Loss: {:.4f}".format(np.mean(train_losses), metrics['loss']))

    tuning_metric = metrics["loss"]
    scheduler.step(tuning_metric)
    is_improvement = tuning_metric > best_metric
    if is_improvement:
        best_metric = tuning_metric
        n_no_improve = 0
    else:
        n_no_improve += 1

    save_checkpoint(
        {
            "epoch": i_epoch + 1,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "n_no_improve": n_no_improve,
            "best_metric": best_metric,
        },
        is_improvement,
        args.savedir,
    )

    if n_no_improve >= args.patience:
        print("No improvement. Breaking out of loop.")
        break


load_checkpoint(model, os.path.join(args.savedir, f"{EPOCHS}checkpoint.pt"))
model.eval()
preds, test_metrics = model_eval(np.inf, val_loader, model, args, criterion)
decoded_preds = [[id2label[i] for i, l in enumerate(row) if l == 1] for row in preds]

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
jdata = read_dataset_file('/arc/project/st-wangll05-1/amir/concat_bert/dev_subtask2a_en.json')
for d in jdata:
    ids_list.append(d['id'])

transformed_data = transform_to_structure(decoded_preds, ids_list)

output_file_path = f'/scratch/st-wangll05-1/amir/concatBERT/outputs/concatBERT_{EPOCHS}epoch_clip_new_caption2.json'
with open(output_file_path, 'w', encoding='utf-8') as output_file:
    json.dump(transformed_data, output_file, indent=4, ensure_ascii=False)
