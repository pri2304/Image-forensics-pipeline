import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import pandas as pd
from PIL import Image, ImageFile
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, classification_report
import os
import gc

# Handle truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ================= CONFIGURATION =================
CSV_FILE = "final_cnn_dataset.csv"
MODEL_SAVE_PATH = "efficientnet_forensics_best.pth"

# MODEL SETTINGS
IMG_SIZE = 380  # Keep High Resolution
BATCH_SIZE = 8  # Physical Batch (Fits in VRAM)
ACCUM_STEPS = 4  # Virtual Batch Multiplier (8 * 4 = 32 Effective Batch)
EPOCHS = 15
LEARNING_RATE = 0.0001
NUM_WORKERS = 8

# OVERSAMPLING RULES
OVERSAMPLE_CONFIG = {
    "FlickrFace": 5,
    "FaceForensics_Real": 5,
    "FantasyID_Real": 5,
    "Smartphone": 4,
    "Social_media_laundered_real": 4,
    "Social_media_laundered_fake": 3
}


# =================================================

class ForensicDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.data = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = row['path']
        label = int(row['label'])
        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, torch.tensor(label, dtype=torch.float32)
        except Exception:
            return torch.zeros((3, IMG_SIZE, IMG_SIZE)), torch.tensor(label, dtype=torch.float32)


class ForensicEvalDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.data = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = row['path']
        label = int(row['label'])
        tag = row['tag1']
        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, label, tag
        except:
            return torch.zeros((3, IMG_SIZE, IMG_SIZE)), label, tag


def oversample_dataframe(df):
    print("\n--- Applying In-Memory Oversampling ---")
    new_rows = []
    for index, row in df.iterrows():
        tag = str(row.get('tag1', ''))
        multiplier = 1
        for key, val in OVERSAMPLE_CONFIG.items():
            if key in tag:
                multiplier = val
                break
        for _ in range(multiplier):
            new_rows.append(row)
    return pd.DataFrame(new_rows).sample(frac=1, random_state=42).reset_index(drop=True)


def train_and_eval():
    # Force clean memory before start
    torch.cuda.empty_cache()
    gc.collect()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Upgrading to EfficientNet-B4 on {device} ---")
    print(f"--- Strategy: Batch Size {BATCH_SIZE} x {ACCUM_STEPS} Steps = Effective {BATCH_SIZE * ACCUM_STEPS} ---")

    # 1. LOAD & SPLIT
    df = pd.read_csv(CSV_FILE)
    try:
        train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['tag1'], random_state=42)
    except:
        train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

    train_df = oversample_dataframe(train_df)

    # 2. TRANSFORMS (380px)
    train_transforms = transforms.Compose([
        transforms.Resize((400, 400)),
        transforms.RandomCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 3. LOADERS
    train_loader = DataLoader(ForensicDataset(train_df, train_transforms), batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader_simple = DataLoader(ForensicDataset(val_df, val_transforms), batch_size=BATCH_SIZE, shuffle=False,
                                   num_workers=NUM_WORKERS, pin_memory=True)

    # 4. MODEL SETUP
    print("Initializing EfficientNet_B4...")
    model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.amp.GradScaler('cuda')

    # 5. TRAINING LOOP (WITH ACCUMULATION)
    best_recall = 0.0
    optimizer.zero_grad()  # Initialize gradients

    for epoch in range(EPOCHS):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")

        for i, (images, labels) in enumerate(loop):
            images, labels = images.to(device), labels.to(device).unsqueeze(1)

            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
                # Normalize loss to account for accumulation
                loss = loss / ACCUM_STEPS

            scaler.scale(loss).backward()

            # Apply updates only every ACCUM_STEPS
            if (i + 1) % ACCUM_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                # Show non-normalized loss in progress bar for readability
                loop.set_postfix(loss=loss.item() * ACCUM_STEPS)

            # Handle remainder batch at end of epoch
            if (i + 1) == len(train_loader) and (i + 1) % ACCUM_STEPS != 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        # Validation
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for images, labels in val_loader_simple:
                images, labels = images.to(device), labels.to(device).unsqueeze(1)
                outputs = model(images)
                preds = torch.sigmoid(outputs) > 0.5
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())

        rec = recall_score(val_targets, val_preds, zero_division=0)
        acc = accuracy_score(val_targets, val_preds)
        print(f"Val Acc: {acc:.4f} | Recall: {rec:.4f}")

        if rec >= best_recall:
            best_recall = rec
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

    print("\n--- Generating Final Report ---")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.eval()

    eval_loader = DataLoader(ForensicEvalDataset(val_df, val_transforms), batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS)
    all_preds, all_labels, all_tags = [], [], []

    with torch.no_grad():
        for images, labels, tags in tqdm(eval_loader):
            images = images.to(device)
            outputs = model(images)
            preds = torch.sigmoid(outputs).cpu().numpy() > 0.5
            all_preds.extend(preds.flatten())
            all_labels.extend(labels.numpy())
            all_tags.extend(tags)

    results = pd.DataFrame({"Tag": all_tags, "True": all_labels, "Pred": all_preds})
    results["Correct"] = results["True"] == results["Pred"]
    report = results.groupby("Tag").agg(Count=('Correct', 'count'), Accuracy=('Correct', 'mean')).sort_values(
        by="Accuracy")

    print("\n" + "=" * 60)
    print(f"{'CATEGORY':<30} | {'CNT':<5} | {'ACCURACY':<8} | {'STATUS'}")
    print("=" * 60)
    for tag, row in report.iterrows():
        acc = row['Accuracy'] * 100
        status = "✅" if acc > 80 else "⚠️" if acc > 60 else "❌"
        print(f"{tag:<30} | {row['Count']:<5} | {acc:.1f}%    | {status}")
    print("=" * 60)


if __name__ == "__main__":
    train_and_eval()