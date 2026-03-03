# midtermcode.py
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

# -----------------------------
# 1️⃣ Load the dataset
# -----------------------------
data = pd.read_csv("data/jets_games.csv")  # Make sure this path is correct
print("Dataset loaded. Shape:", data.shape)
print(data.head())

# Assume target column is named 'winner'
X = data.drop(columns=["winner"])
y = data["winner"]

# -----------------------------
# 2️⃣ Split into train/val/test
# -----------------------------
# 80% train+val, 20% test
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

# 75% train, 25% val from temp → 60% train, 20% val overall
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42, shuffle=True
)

print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# -----------------------------
# 3️⃣ Convert to PyTorch tensors
# -----------------------------
X_train = torch.tensor(X_train.values, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.long)
X_val = torch.tensor(X_val.values, dtype=torch.float32)
y_val = torch.tensor(y_val.values, dtype=torch.long)
X_test = torch.tensor(X_test.values, dtype=torch.float32)
y_test = torch.tensor(y_test.values, dtype=torch.long)

train_ds = TensorDataset(X_train, y_train)
val_ds = TensorDataset(X_val, y_val)
test_ds = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=16)
test_loader = DataLoader(test_ds, batch_size=16)

# -----------------------------
# 4️⃣ Define the model with Dropout
# -----------------------------
class Net(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3),          # Dropout to reduce overfitting
            nn.Linear(32, output_dim)
        )
    def forward(self, x):
        return self.fc(x)

model = Net(X_train.shape[1], len(y.unique()))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)  # weight_decay for regularization

# -----------------------------
# 5️⃣ Train with validation
# -----------------------------
for epoch in range(1, 101):
    model.train()
    running_loss = 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # Validation loss
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            out = model(xb)
            val_loss += criterion(out, yb).item()

    if epoch % 10 == 0:
        print(f"Epoch [{epoch}/100] | Train Loss: {running_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f}")

# -----------------------------
# 6️⃣ Evaluate on test set
# -----------------------------
correct = 0
total = 0
model.eval()
with torch.no_grad():
    for xb, yb in test_loader:
        out = model(xb)
        pred = torch.argmax(out, dim=1)
        correct += (pred == yb).sum().item()
        total += yb.size(0)

print(f"Test Accuracy: {100*correct/total:.2f}%")