import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

BATCH_SIZE = 8
IMG_SIZE = 64
EPOCHS = 3


parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, required=True)
args = parser.parse_args()

DATASET_NAME = args.data.split("/")[-1]   # e.g. data04
DATA_DIR = f"{args.data}/train"

# Transform to tensors
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

# Load dataset
dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)

num_classes = len(dataset.classes)
print("Classes:", dataset.classes)

# Split into train/test
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# Models
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * IMG_SIZE * IMG_SIZE, num_classes)
        )

    def forward(self, x):
        return self.net(x)


class DeeperCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * IMG_SIZE * IMG_SIZE, num_classes)
        )

    def forward(self, x):
        return self.net(x)


# Train
def train_model(model, loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for _ in range(EPOCHS):
        for X, y in loader:
            outputs = model(X)
            loss = criterion(outputs, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


# Eval
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for X, y in loader:
            outputs = model(X)
            _, preds = torch.max(outputs, 1)

            correct += (preds == y).sum().item()
            total += y.size(0)

    return correct / total


# AutoML loop
models = {
    "simple": SimpleCNN(),
    "deep": DeeperCNN()
}

best_model = None
best_score = 0
best_name = ""

for name, model in models.items():
    print(f"\nTraining: {name}")

    train_model(model, train_loader)
    score = evaluate(model, test_loader)

    print(f"{name} accuracy: {score:.4f}")

    if score > best_score:
        best_score = score
        best_model = model
        best_name = name

print(f"\nBest Model: {best_name} ({best_score:.4f})")

# Save model
model_filename = f"{DATASET_NAME}_best_{best_name}.pt"
torch.save(best_model.state_dict(), model_filename)
print(f"Saved {model_filename}")
print(f"MODEL_NAME={model_filename}")