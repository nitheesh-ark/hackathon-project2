import torch
from model import MetaSurplusModel

menu_map = {
    0: ["Pasta Alfredo", "Spaghetti Bolognese", "Margherita Pizza", "Garlic Bread"],
    1: ["Hakka Noodles", "Fried Rice", "Gobi Manchurian", "Chilli Paneer"],
    2: ["Steamed Rice", "Sambar", "Rasam", "Poriyal"],
    3: ["Butter Chicken", "Paneer Butter Masala", "Dal Tadka", "Jeera Rice"]
}

# -------------------------
# Load Model
# -------------------------
model = MetaSurplusModel()
model.load_state_dict(torch.load("meta_model_v2.pth", map_location="cpu"))
model.eval()

# -------------------------
# Predict
# -------------------------
def predict(footfall, food_prepared, day, menu):

    x = torch.tensor([[
        footfall / 1000.0,
        food_prepared / 1000.0,
        day / 6.0,
        menu / 3.0
    ]], dtype=torch.float32)

    with torch.no_grad():
        pred = model(x)[0].numpy()

    # 🔥 Rescale back
    pred = pred * 200.0

    dishes = menu_map[menu]

    print("\n🍽️ Predicted Surplus:")
    for i in range(4):
        print(f"{dishes[i]} → {pred[i]:.1f} plates")



if __name__ == "__main__":
    predict(420, 500, 6, 1)