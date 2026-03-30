import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from model import MetaSurplusModel

# -------------------------
# Synthetic Data
# -------------------------
np.random.seed(42)

n = 10000

data = pd.DataFrame({
    "footfall_pred": np.random.randint(200, 900, n),
    "food_prepared": np.random.randint(250, 800, n),
    "day": np.random.randint(0, 7, n),
    "menu": np.random.randint(0, 4, n)
})

# -------------------------
# Normalize Inputs
# -------------------------
data["footfall_pred"] /= 1000.0
data["food_prepared"] /= 1000.0
data["day"] /= 6.0
data["menu"] /= 3.0

# -------------------------
# Generate Targets
# -------------------------

targets = []

for i in range(n):
    base = (data["food_prepared"][i] - data["footfall_pred"][i]) * 1000

    dishes = []
    for d in range(4):
        noise = np.random.normal(0, 2)  # 🔥 reduced noise

        day_factor = 10 if data["day"][i] > 0.7 else 0
        menu_factor = data["menu"][i] * 5

        value = base / 4 + noise + day_factor + menu_factor
        dishes.append(value)

    targets.append(dishes)
    print(targets[-1])


X = torch.tensor(data.values, dtype=torch.float32)
y = torch.tensor(targets, dtype=torch.float32)



y = y / 200.0

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#---------------------------------------------------------------------------------------------------------------------------------------------


model = MetaSurplusModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()



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
# Save Model
# -------------------------
torch.save(model.state_dict(), "meta_model_v2.pth")

print("✅ Model trained and saved as meta_model_v2.pth")