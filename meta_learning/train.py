import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from model import MetaSurplusModel

# ---------- SAMPLE DATA ----------
import numpy as np

np.random.seed(42)

n = 1000
data = pd.DataFrame({
    "footfall_pred": np.random.randint(200, 900, n),
    "food_prepared": np.random.randint(250, 800, n),
    "day": np.random.randint(0, 7, n),
    "menu": np.random.randint(0, 5, n)
})


# simulated surplus
data["surplus"] = (
    data["food_prepared"] - data["footfall_pred"]
    + data["menu"] * 5
    - data["day"] * 5
    + np.random.normal(0, 10, n)
)
print(data)

# ---------- PREP ----------
X = data[["footfall_pred", "food_prepared", "day", "menu"]].values
y = data["surplus"].values

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# ---------- MODEL ----------
model = MetaSurplusModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.MSELoss()

# ---------- TRAIN ----------
for epoch in range(500):
    optimizer.zero_grad()

    preds = model(X_train)
    loss = loss_fn(preds, y_train)

    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item():.2f}")

# ---------- SAVE ----------
torch.save(model.state_dict(), "meta_model.pth")

print("Model saved as meta_model.pth")