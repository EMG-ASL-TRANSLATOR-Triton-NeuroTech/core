import re
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path


# ----------------------------
# 1. Extracts data from CSV
# ----------------------------
class EMGDataset(Dataset):
    """
    Loads 169×4 EMG CSV files → reshaped to (4, 13, 13) tensors
    This program expects the file name to be written as gesture1_sample1.csv
    The first row of the csv should be the column headings, and the rest of the rows should be the data.
    The type of gesture it represents is determined by its file name.
    For all training data, it should be placed in one file only.
    The program will automatically process the data and put them into tensors.
    """

    def __init__(self, data_dir):
        self.samples = []
        self.labels = []
        self.gesture_map = {}  # gesture_id → class_index

        # Extract gesture IDs from filenames and build mapping
        csv_files = sorted(Path(data_dir).glob("*.csv"))
        gesture_ids = sorted({int(re.search(r'gesture(\d+)', f.stem).group(1)) for f in csv_files})
        self.gesture_map = {gid: idx for idx, gid in enumerate(gesture_ids)}

        # Load data
        for csv_path in csv_files:
            gesture_id = int(re.search(r'gesture(\d+)', csv_path.stem).group(1))
            label = self.gesture_map[gesture_id]

            # Load CSV → reshape 169×4 → 13×13×4 → (4,13,13)
            emg = pd.read_csv(csv_path).values.reshape(13, 13, 4).transpose(2, 0, 1)
            self.samples.append(torch.tensor(emg, dtype=torch.float32))
            self.labels.append(label)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], torch.tensor(self.labels[idx], dtype=torch.long)


# ----------------------------
# 2. CNN Architecture
# ----------------------------
class EMGNet(nn.Module):
    """
    3 Conv + 2 FC layers for 4-channel 13×13 EMG input → 5 classes
    Number of classes can be modified for more gestures
    """

    def __init__(self, num_classes=5):
        super().__init__()
        self.features = nn.Sequential(
            # Conv1: (4,13,13) → (16,6,6)
            nn.Conv2d(4, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Conv2: (16,6,6) → (32,3,3)
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Conv3: (32,3,3) → (64,3,3)
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 3 * 3, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ----------------------------
# 3. Training Pipeline
# ----------------------------
def main():
    """
    Configuration here, change the parameters for different cases
    Change batch size/epoch for different learning scenarios
    """
    DATA_DIR = "Processed_CSV"
    NUM_CLASSES = 5
    BATCH_SIZE = 8
    NUM_EPOCHS = 100
    LR = 0.001
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load dataset
    dataset = EMGDataset(DATA_DIR)
    print(f"Loaded {len(dataset)} samples across {NUM_CLASSES} classes | Mapping: {dataset.gesture_map}")

    # Train/val split (80/20)
    train_size = int(0.8 * len(dataset))

    """
    Need to modify training and validation dataset sizes when there's more datasets
    """
    train_ds, val_ds = random_split(dataset, [train_size, len(dataset) - train_size],
                                    generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = EMGNet(NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Training
    best_acc = 0.0
    for epoch in range(NUM_EPOCHS):
        # Train
        model.train()
        train_loss, correct, total = 0, 0, 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, pred = out.max(1)
            total += y.size(0)
            correct += pred.eq(y).sum().item()

        # Validate
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                out = model(x)
                _, pred = out.max(1)
                val_total += y.size(0)
                val_correct += pred.eq(y).sum().item()
        train_acc = 100. * correct / total
        val_acc = 100. * val_correct / val_total
        print(f"Epoch {epoch + 1:3d}/{NUM_EPOCHS} | "
              f"Train Loss: {train_loss / len(train_loader):.4f} Acc: {train_acc:6.2f}% | "
              f"Val Acc: {val_acc:6.2f}%")

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), '../best_emg_model.pth')

    torch.save(model.state_dict(), '../emg_gesture_model_final.pth')
    print("trained on", DEVICE)
    print(f"\nTraining complete. Best validation accuracy: {best_acc:.2f}%")
    print("Models saved: 'best_emg_model.pth' (best) | 'emg_gesture_model_final.pth' (final)")


if __name__ == "__main__":
    main()