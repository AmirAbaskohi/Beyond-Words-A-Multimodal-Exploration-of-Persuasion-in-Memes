from torch.utils.data import Dataset, DataLoader
from datasets import load_from_disk
from PIL import Image
import io
from transformers import AutoProcessor, Blip2ForConditionalGeneration
from peft import LoraConfig, get_peft_model
import torch
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
import os

dataset_path = 'BLIP_MemeCap_Dataset/train'
dataset = load_from_disk(dataset_path)

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


config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "k_proj"]
)

model = get_peft_model(model, config)
model.print_trainable_parameters()


train_dataset = ImageCaptioningDataset(dataset, processor)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=8, collate_fn=collate_fn)


device = "cuda" if torch.cuda.is_available() else "cpu"
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

checkpoint_dir = "BLIP2_checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

num_epochs = 1
num_steps_to_save = 100
start_epoch = 0
start_step = 0

# Uncomment the following lines if you saved the optimizer state
# optimizer_state_dict = torch.load("path/to/optimizer_state_dict.pth")
# optimizer.load_state_dict(optimizer_state_dict)

loss_file_path = "BLIP2_checkpoints/loss.txt"
with open(loss_file_path, "w") as loss_file:
  loss_file.write("Epoch,Step,Loss\n") 
  for epoch in range(start_epoch, num_epochs):
      for idx, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch}")):
          if epoch == start_epoch and idx < start_step:
              continue

          input_ids = batch.pop("input_ids").to(device)#, dtype=torch.float16)
          pixel_values = batch.pop("pixel_values").to(device)#, dtype=torch.float16)

          outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=input_ids)

          loss = outputs.loss

          print(f"Epoch {epoch}, Step {idx + 1}, Loss: {loss.item()}")

          loss_file.write(f"{epoch},{idx + 1},{loss.item()}\n")

          loss.backward()

          optimizer.step()
          optimizer.zero_grad()

          if (idx + 1) % num_steps_to_save == 0:
              checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch}_step_{idx + 1}")
              model.save_pretrained(checkpoint_path)
              torch.save(optimizer.state_dict(), os.path.join(checkpoint_path, "optimizer_state_dict.pth"))


# model.push_to_hub("BCAmirs/blip2-opt-6.7b-MemeCap")