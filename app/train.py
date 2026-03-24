import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# =====================
# Data Loading
# =====================
X, y = load_iris(return_X_y=True)

scaler = StandardScaler()
X = scaler.fit_transform(X)

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =====================
# Models
# =====================

class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 3)

    def forward(self, x):
        return self.fc(x)


class DeepModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 3)
        )

    def forward(self, x):
        return self.net(x)


class BiggerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 3)
        )

    def forward(self, x):
        return self.net(x)


# =====================
# Train Function
# =====================
def train_model(model, X_train, y_train, epochs=50):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(epochs):
        model.train()

        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# =====================
# Evaluation Function
# =====================
def evaluate(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        _, preds = torch.max(outputs, 1)
        accuracy = (preds == y_test).float().mean()

    return accuracy.item()


# =====================
# AutoML Loop
# =====================
models = {
    "linear": LinearModel(),
    "deep": DeepModel(),
    "bigger": BiggerModel()
}

best_model = None
best_score = 0
best_name = ""

for name, model in models.items():
    print(f"\nTraining: {name}")

    train_model(model, X_train, y_train)
    score = evaluate(model, X_test, y_test)

    print(f"{name} accuracy: {score:.4f}")

    if score > best_score:
        best_score = score
        best_model = model
        best_name = name

print(f"\nBest Model: {best_name} ({best_score:.4f})")

# =====================
# Save Best Model
# =====================
torch.save(best_model.state_dict(), "best_model.pt")

print("Saved best_model.pt")