from torch.utils.data import Dataset, DataLoader
from datasets import load_from_disk
from PIL import Image
import io
from transformers import Blip2ForConditionalGeneration, AutoProcessor
from peft import PeftModel, PeftConfig, LoraConfig, get_peft_model
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
import os

dataset_test_path = 'BLIP_MemeCap_Dataset/test'
dataset_test = load_from_disk(dataset_test_path)

class ImageCaptioningDataset(Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = Image.open(io.BytesIO(item["image"]["bytes"]))
        encoding = self.processor(images=image, padding="max_length", return_tensors="pt")
        encoding = {k: v.squeeze() for k, v in encoding.items()}
        encoding["text"] = item["text"]
        return encoding

def collate_fn(batch):
    processed_batch = {}
    for key in batch[0].keys():
        if key != "text":
            processed_batch[key] = torch.stack([example[key] for example in batch])
        else:
            text_inputs = processor.tokenizer(
                [example["text"] for example in batch], padding=True, return_tensors="pt"
            )
            processed_batch["input_ids"] = text_inputs["input_ids"]
            processed_batch["attention_mask"] = text_inputs["attention_mask"]
    return processed_batch


processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-6.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-6.7b", device_map="auto", load_in_8bit=True)#, torch_dtype=torch.float16)

peft_model_id = PATH_TO_YOUR_SAVED_MODEL
config = PeftConfig.from_pretrained(peft_model_id)

model = Blip2ForConditionalGeneration.from_pretrained(config.base_model_name_or_path, device_map="auto", load_in_8bit=True)
model = PeftModel.from_pretrained(model, peft_model_id)

batch_size = 8
predicts = []
device = "cuda" if torch.cuda.is_available() else "cpu"

batched_examples = [dataset_test[i:i + batch_size] for i in range(0, len(dataset_test), batch_size)]

for batch_examples in tqdm(batched_examples, desc="Processing Batches"):
    images = [
        Image.open(io.BytesIO(example["bytes"])) for example in batch_examples["image"]
    ]

    inputs = processor(images=images, return_tensors="pt", padding=True).to(device, torch.float16)
    pixel_values = inputs.pixel_values

    generated_ids = model.generate(pixel_values=pixel_values, max_length=100)
    generated_captions = processor.batch_decode(generated_ids, skip_special_tokens=True)

    predicts.extend(generated_captions)

len(predicts)

file_path = "BLIP-finetuned.txt"

with open(file_path, "w") as file:
    for line in predicts:
        file.write(line + "\n")

