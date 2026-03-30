import torch
from .model import FoodPrepModel

menu_map = {
    0: ["Pasta Alfredo", "Spaghetti Bolognese", "Margherita Pizza", "Garlic Bread"],
    1: ["Hakka Noodles", "Fried Rice", "Gobi Manchurian", "Chilli Paneer"],
    2: ["Steamed Rice", "Sambar", "Rasam", "Poriyal"],
    3: ["Butter Chicken", "Paneer Butter Masala", "Dal Tadka", "Jeera Rice"]
}


def predict(footfall, day, menu, model_path = "prep_model.pth"):

    model = FoodPrepModel()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    x = torch.tensor([[
        footfall / 1000.0,
        day / 6.0,
        menu / 3.0
    ]], dtype=torch.float32)

    with torch.no_grad():
        pred = model(x)[0].numpy()

    # 🔥 rescale
    pred = pred * 1000.0


    dishes = menu_map[menu]

    print("\n🍳 Food to Prepare:")

    
    return pred


if __name__ == "__main__":
    predict(
        footfall=500,
        day=6,
        menu=2
    )