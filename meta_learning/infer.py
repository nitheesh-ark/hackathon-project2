import torch
from .model import MetaSurplusModel

menu_map = {
    0: ["Pasta Alfredo", "Spaghetti Bolognese", "Margherita Pizza", "Garlic Bread"],
    1: ["Hakka Noodles", "Fried Rice", "Gobi Manchurian", "Chilli Paneer"],
    2: ["Steamed Rice", "Sambar", "Rasam", "Poriyal"],
    3: ["Butter Chicken", "Paneer Butter Masala", "Dal Tadka", "Jeera Rice"]
}




def predict(footfall, food_prepared, day, menu, model_path = "meta_model_v2.pth"):

    model = MetaSurplusModel()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()


    x = torch.tensor([[
        footfall / 1000.0,
        food_prepared / 1000.0,
        day / 6.0,
        menu / 3.0
    ]], dtype=torch.float32)

    with torch.no_grad():
        pred = model(x)[0].numpy()


    pred = pred * 200.0

    dishes = menu_map[menu]

    print("\nPredicted Surplus:")

    return pred
        
    



if __name__ == "__main__":
    predict(500, 620, 6, 3)