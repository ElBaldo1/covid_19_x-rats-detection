# covid.ipynb

# ============================================================
# 1. Import Required Libraries
# ============================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torchvision import datasets, transforms
from torchvision.utils import make_grid

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import random
import os
import csv

# For TensorBoard (optional, if logging is needed)
# Run: pip install tensorboard
# then: tensorboard --logdir=runs
from torch.utils.tensorboard import SummaryWriter


# ============================================================
# 2. Class Mapping (label -> string)
# ============================================================
class_mapping = {
    0: "Benigne",
    1: "Malignant",
}


def get_class(label):
    """
    label: tensor with class index (0, 1, ...)
    Returns the corresponding string (e.g., 'Benigne', 'Malignant').
    """
    return class_mapping[label.item()]


# ============================================================
# 3. Custom Dataset Creation
#    (CTDataset defined in import_Data.py)
# ============================================================
try:
    from import_Data import CTDataset
except ImportError:
    print("Ensure you have the import_Data.py file with the CTDataset class.")

# ============================================================
# 4. Set Random Seed for Reproducibility
# ============================================================
SEED = 1
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

# ============================================================
# 5. Image Transformations (Advanced Data Augmentation)
# ============================================================
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ============================================================
# 6. Full Dataset Creation
# ============================================================
root = 'COVID-Data-Radiography'
dataset = CTDataset(root, transform=transform, num_classes=2)

# ============================================================
# 7. Split Dataset into Train / Validation / Test
# ============================================================
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_db, val_db, test_db = random_split(dataset, [train_size, val_size, test_size])

print(f"Train set size: {len(train_db)}")
print(f"Validation set size: {len(val_db)}")
print(f"Test set size: {len(test_db)}")

# ============================================================
# 8. Handling Imbalanced Classes
# ============================================================
all_labels = [train_db[i][1] for i in range(len(train_db))]
class_count = np.bincount(all_labels)
print("Class count (Train set):", class_count)

class_weights = 1.0 / class_count
sample_weights = [class_weights[label] for label in all_labels]

train_sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

# ============================================================
# 9. DataLoader Setup
# ============================================================
batch_size = 8

train_loader = DataLoader(train_db, batch_size=batch_size, sampler=train_sampler, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_db, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_db, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

# ============================================================
# 10. Visualize a Batch
# ============================================================
def visualize_batch(dataloader):
    dataiter = iter(dataloader)
    images, labels = next(dataiter)
    ncols = min(batch_size, 4)
    nrows = int(np.ceil(len(images) / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(15, nrows * 3))
    axes = axes.flatten() if nrows * ncols > 1 else [axes]

    for i in range(len(images)):
        ax = axes[i]
        img = images[i].permute(1, 2, 0).numpy()
        ax.imshow(img[:, :, 0], cmap='gray')
        ax.axis('off')
        ax.set_title(get_class(labels[i]), color='red')

    for i in range(len(images), len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

# ============================================================
# 11. Define the Neural Network (CNN)
# ============================================================
class RadiographyCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(RadiographyCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(64, 32, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(32, 32, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(32)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(32 * 26 * 26, 1024)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)

# ============================================================
# 12. Instantiate the Model and Move to GPU if Available
# ============================================================
model = RadiographyCNN(num_classes=2)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")
model.to(device)

# ============================================================
# 13. Define Loss Function and Optimizer
# ============================================================
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# ============================================================
# 14. Learning Rate Scheduler
# ============================================================
from torch.optim.lr_scheduler import ReduceLROnPlateau

scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, )

# ============================================================
# 15. Training and Evaluation Functions
# ============================================================
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return running_loss / len(dataloader), correct / total


def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

    return running_loss / len(dataloader), correct / total

# ============================================================
# 16. Early Stopping and Best Model Saving
# ============================================================
early_stop_patience = 3
best_val_loss = float('inf')
epochs_no_improve = 0
best_model_path = 'best_model.pth'

# ============================================================
# 17. TensorBoard and CSV Logging
# ============================================================
writer = SummaryWriter(log_dir='runs/COVID_experiment')

csv_file = 'training_log.csv'
with open(csv_file, 'w', newline='') as f:
    csv.writer(f).writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'lr'])

# ============================================================
# 18. Training Loop with Validation
# ============================================================
if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method('spawn')

    visualize_batch(train_loader)

    epochs = 20
    train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []

    start_time = time.time()

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        # Aggiorna lo scheduler in base al valore della loss di validazione
        scheduler.step(val_loss)
        current_lr = scheduler.get_last_lr()[0]
        print(
            f"[Epoch {epoch + 1}/{epochs}] Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} | LR: {current_lr}")

        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_acc, epoch)
        writer.add_scalar('Accuracy/Validation', val_acc, epoch)
        writer.add_scalar('LR', current_lr, epoch)

        with open(csv_file, 'a', newline='') as f:
            csv.writer(f).writerow([epoch + 1, train_loss, train_acc, val_loss, val_acc, current_lr])

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            print("   --> Model saved (val_loss improved).")
        else:
            epochs_no_improve += 1
            print(f"   --> No improvement in val_loss for {epochs_no_improve} epoch(s).")
            if epochs_no_improve >= early_stop_patience:
                print("\nEarly stopping triggered.")
                break

    end_time = time.time()
    print(f"\nTraining + Validation completed in {(end_time - start_time) / 60:.2f} minutes")

    writer.close()

    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
        print(f"Best model loaded from '{best_model_path}'")

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\n--- Test Results ---\nTest Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()
