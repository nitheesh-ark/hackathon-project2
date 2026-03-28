import torch
from model import MetaSurplusModel
from adapt import adapt_model

# load model
model = MetaSurplusModel()
model.load_state_dict(torch.load("meta_model.pth"))
model.eval()

# sample test input
# [footfall_pred, food_prepared, day, menu]
x = torch.tensor([
    [420, 500, 2, 1],
    [380, 480, 3, 2]
], dtype=torch.float32)

# ---------- BASE PREDICTION ----------
with torch.no_grad():
    pred = model(x)
    print("Base Surplus Prediction:", pred)

# ---------- ADAPTATION ----------
# simulate real observed surplus (like feedback)
y_real = torch.tensor([
    [80.0],
    [120.0]
])

adapted_model = adapt_model(model, x, y_real)

# ---------- AFTER ADAPT ----------
with torch.no_grad():
    new_pred = adapted_model(x)
    print("Adapted Prediction:", new_pred)