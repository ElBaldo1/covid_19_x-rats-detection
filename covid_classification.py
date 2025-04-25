# ============================================================
# 1. Imports
# ============================================================
import os, csv, time, random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix

# ============================================================
# 2. Utilities
# ============================================================
class_mapping = {0: "No Covid", 1: "Yes Covid"}
get_class = lambda lbl: class_mapping[int(lbl)]

try:
    from import_Data import CTDataset  # custom ImageFolder wrapper
except ImportError as e:
    raise ImportError("Missing import_Data.py with CTDataset class") from e

# Reproducibility
SEED = 1
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ============================================================
# 3. Transforms & Dataset
# ============================================================
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
    transforms.Grayscale(1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

dataset = CTDataset("COVID-Data-Radiography", transform=transform)
train_sz = int(0.7 * len(dataset)); val_sz = int(0.15 * len(dataset))
remaining = len(dataset) - train_sz - val_sz
train_db, val_db, test_db = random_split(
    dataset, [train_sz, val_sz, remaining],
    generator=torch.Generator().manual_seed(SEED)
)

# ============================================================
# 4. DataLoaders
# ============================================================
all_labels = [train_db[i][1] for i in range(len(train_db))]
class_weights = 1.0 / np.bincount(all_labels)
sampler = WeightedRandomSampler([class_weights[l] for l in all_labels], len(all_labels))

batch = 8
train_dl = DataLoader(train_db, batch, sampler=sampler, num_workers=2, pin_memory=True)
val_dl   = DataLoader(val_db, batch, shuffle=False, num_workers=2, pin_memory=True)
test_dl  = DataLoader(test_db, batch, shuffle=False, num_workers=2, pin_memory=True)

# ============================================================
# 5. CNN
# ============================================================
class RadiographyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 3)
        self.bn1   = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 32, 3)
        self.bn2   = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, 3)
        self.bn3   = nn.BatchNorm2d(32)
        self.pool  = nn.MaxPool2d(2)
        self.fc1   = nn.Linear(32*26*26, 1024)
        self.drop  = nn.Dropout(0.5)
        self.fc2   = nn.Linear(1024, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.flatten(1)
        x = self.drop(F.relu(self.fc1(x)))
        return self.fc2(x)

# ------------------------------------------------------------
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model   = RadiographyCNN().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
from torch.optim.lr_scheduler import ReduceLROnPlateau
scheduler = ReduceLROnPlateau(optimizer, "min", 0.5, 2)

best_model_path = "best_model.pth"
os.makedirs("outputs", exist_ok=True)

# ============================================================
# 6. Training helpers
# ============================================================

def run_epoch(model, loader, train=True):
    if train: model.train(); optimizer.zero_grad()
    else: model.eval()
    loss_sum = correct = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = criterion(out, y)
        if train:
            loss.backward(); optimizer.step(); optimizer.zero_grad()
        loss_sum += loss.item(); correct += (out.argmax(1)==y).sum().item()
    return loss_sum/len(loader), correct/len(loader.dataset)

# ============================================================
# 7. Detailed evaluation & logging
# ============================================================
def evaluate_detailed(model, loader):
    model.eval(); preds, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            preds.extend(model(x.to(DEVICE)).argmax(1).cpu().numpy())
            labels.extend(y.numpy())
    report = classification_report(labels, preds, target_names=list(class_mapping.values()))
    print("\nClassification report:\n", report)
    with open("outputs/classification_report.txt", "w") as f:
        f.write(report)
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(4,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=list(class_mapping.values()),
                yticklabels=list(class_mapping.values()))
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("Confusion Matrix"); plt.tight_layout()
    plt.savefig("outputs/cm.png", dpi=300)
    plt.close()

# ============================================================
# 8. Main
# ============================================================
if __name__ == "__main__":
    # Skip training if weights exist
    if os.path.exists(best_model_path):
        print("Found existing model weights, skipping training.")
        model.load_state_dict(torch.load(
            best_model_path,
            map_location = DEVICE,
            weights_only = True
                                ))
    else:
        # prepare CSV for logging
        csv_path = os.path.join("outputs", "training_log.csv")
        with open(csv_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"])

            epochs, best_val, patience = 20, float("inf"), 3
            no_imp = 0
            train_losses, val_losses = [], []
            train_accs, val_accs = [], []

            for ep in range(1, epochs+1):
                tl, ta = run_epoch(model, train_dl, True)
                vl, va = run_epoch(model, val_dl, False)
                scheduler.step(vl)
                train_losses.append(tl); val_losses.append(vl)
                train_accs.append(ta); val_accs.append(va)
                print(f"Ep{ep}: TL={tl:.4f} TA={ta:.4f} | VL={vl:.4f} VA={va:.4f}")

                # log to CSV
                writer.writerow([ep, f"{tl:.4f}", f"{ta:.4f}", f"{vl:.4f}", f"{va:.4f}"])

                if vl < best_val:
                    best_val, no_imp = vl, 0
                    torch.save(model.state_dict(), best_model_path)
                else:
                    no_imp += 1
                    if no_imp >= patience:
                        print("Early stopping")
                        break

        # Plot & save training curves
        plt.figure(figsize=(12,5))
        plt.subplot(1,2,1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.title('Loss vs Epoch'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
        plt.subplot(1,2,2)
        plt.plot(train_accs, label='Train Acc')
        plt.plot(val_accs, label='Val Acc')
        plt.title('Accuracy vs Epoch'); plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend()
        plt.tight_layout()
        plt.savefig("outputs/training_curves.png", dpi=300)
        plt.close()

    # Evaluate final model
    test_loss, test_acc = run_epoch(model, test_dl, False)
    print(f"Test Loss={test_loss:.4f}, Test Acc={test_acc:.4f}")
    evaluate_detailed(model, test_dl)

