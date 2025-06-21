import os
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel, get_scheduler
from torch.optim import AdamW


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ----------------------------
# PATHS
# ----------------------------
BASE_DIR = "/kaggle/input/real-fake-product-dataset/FAKE-PRODUCT-DETECTION-SYSTEM-1"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
VALID_DIR = os.path.join(BASE_DIR, "valid")
TEST_DIR  = os.path.join(BASE_DIR, "test")

TRAIN_CSV = os.path.join(TRAIN_DIR, "_classes.csv")
VALID_CSV = os.path.join(VALID_DIR, "_classes.csv")
TEST_CSV  = os.path.join(TEST_DIR, "_classes.csv")

# ----------------------------
# Load DataFrames and Convert to Real/Fake Binary Labels
# ----------------------------
def preprocess_labels(csv_path, image_dir):
    df = pd.read_csv(csv_path)
    label_cols = ['ADIDAS_fake', 'ADIDAS_original', 'FILA_fake', 
                  'FILA_original', 'NIKE_fake', 'NIKE_original', 
                  'PUMA_fake', 'PUMA_original']
    
    def label_mapper(row):
        for col in label_cols:
            if row[col] == 1:
                return 1 if "original" in col else 0
        return -1  # should never happen

    df["label"] = df.apply(label_mapper, axis=1)
    df["filepath"] = df["filename"].apply(lambda x: os.path.join(image_dir, x))
    return df[["filepath", "label"]]

df_train = preprocess_labels(TRAIN_CSV, TRAIN_DIR)
df_valid = preprocess_labels(VALID_CSV, VALID_DIR)
df_test  = preprocess_labels(TEST_CSV, TEST_DIR)

# ----------------------------
# Dataset and Dataloader
# ----------------------------
class ProductDataset(Dataset):
    def __init__(self, dataframe, processor):
        self.data = dataframe.reset_index(drop=True)
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = Image.open(self.data.loc[idx, "filepath"]).convert("RGB")
        label = self.data.loc[idx, "label"]
        text = "real product" if label == 1 else "fake product"

        # Prepare a single text prompt per image
        inputs = self.processor(
            text=text,
            images=image,
            return_tensors="pt",
            padding=True
        )

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }


# ----------------------------
# Load CLIP model and processor
# ----------------------------
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

model = model.to(device)

# ----------------------------
# Wrap model with classification head
# ----------------------------
class CLIPClassifier(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.clip = base_model
        self.classifier = nn.Linear(base_model.config.projection_dim, 2)

    def forward(self, input_ids, attention_mask, pixel_values):
        outputs = self.clip(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
            pixel_values=pixel_values.to(device),
            return_loss=False,
        )
        # outputs.logits_per_image: [batch_size, 2] similarity logits
        img_embeds = outputs.image_embeds  # [B, D]
        return self.classifier(img_embeds)

model = CLIPClassifier(model).to(device)

# ----------------------------
# Prepare data loaders
# ----------------------------
train_dataset = ProductDataset(df_train, processor)
valid_dataset = ProductDataset(df_valid, processor)
test_dataset  = ProductDataset(df_test, processor)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=16)
test_loader  = DataLoader(test_dataset, batch_size=16)

# ----------------------------
# Training Setup
# ----------------------------
optimizer = AdamW(model.parameters(), lr=5e-5)
lr_scheduler = get_scheduler("linear", optimizer=optimizer,
                             num_warmup_steps=0, num_training_steps=len(train_loader)*5)
loss_fn = nn.CrossEntropyLoss()

# ----------------------------
# Training Loop
# ----------------------------
def train_model(model, train_loader, valid_loader, epochs=5):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            outputs = model(batch["input_ids"], batch["attention_mask"], batch["pixel_values"])
            #print(outputs.shape)
            loss = loss_fn(outputs, batch["label"].to(device))
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} training loss: {total_loss/len(train_loader):.4f}")
        validate_model(model, valid_loader)

def validate_model(model, valid_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in valid_loader:
            outputs = model(batch["input_ids"], batch["attention_mask"], batch["pixel_values"])
            #print(outputs.shape)
            preds = outputs.argmax(dim=1)
            correct += (preds == batch["label"].to(device)).sum().item()
            total += len(preds)
    print(f"Validation Accuracy: {correct/total:.4f}")

# ----------------------------
# Evaluation
# ----------------------------
def evaluate(model, test_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in test_loader:
            outputs = model(batch["input_ids"], batch["attention_mask"], batch["pixel_values"])
            #print(outputs.shape)
            preds = outputs.argmax(dim=1)
            correct += (preds == batch["label"].to(device)).sum().item()
            total += len(preds)
    print(f"Test Accuracy: {correct/total:.4f}")

# ----------------------------
# Run Everything
# ----------------------------
train_model(model, train_loader, valid_loader, epochs=100)
evaluate(model, test_loader)