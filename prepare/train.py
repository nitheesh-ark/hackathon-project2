import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from model import FoodPrepModel

np.random.seed(42)
n = 2000

# -------------------------
# Inputs
# -------------------------
data = pd.DataFrame({
    "footfall_pred": np.random.randint(200, 900, n),
    "day": np.random.randint(0, 7, n),
    "menu": np.random.randint(0, 4, n)
})

# -------------------------
# Normalize Inputs
# -------------------------
data["footfall_pred"] /= 1000.0
data["day"] /= 6.0
data["menu"] /= 3.0

# -------------------------
# Targets (food to prepare)
# -------------------------
targets = []

for i in range(n):
    footfall = data["footfall_pred"][i] * 1000

    dishes = []

    for d in range(4):
        noise = np.random.normal(0, 3)

        # weekend boost
        day_factor = 20 if data["day"][i] > 0.7 else 0

        # menu popularity
        menu_factor = data["menu"][i] * 10

        # distribute footfall across dishes
        value = (footfall / 4) + day_factor + menu_factor + noise

        dishes.append(value)

    targets.append(dishes)

# -------------------------
# Tensor
# -------------------------
X = torch.tensor(data.values, dtype=torch.float32)
y = torch.tensor(targets, dtype=torch.float32)

# 🔥 Normalize target
y = y / 1000.0

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# -------------------------
# Model
# -------------------------
model = FoodPrepModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

# -------------------------
# Train
# -------------------------
for epoch in range(500):
    model.train()

    optimizer.zero_grad()
    preds = model(X_train)

    loss = loss_fn(preds, y_train)

    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

# -------------------------
# Save
# -------------------------
torch.save(model.state_dict(), "prep_model.pth")

print("✅ Prep model saved!")