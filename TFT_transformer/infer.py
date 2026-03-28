import torch
from model import TFTModel

def predict(sample, model_path="tft_model.pth"):
    seq, day, menu, _ = sample  


    seq = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    day = torch.tensor([day])
    menu = torch.tensor([menu])

    model = TFTModel()
    checkpoint = torch.load(model_path, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])
    mean = checkpoint["mean"]
    std = checkpoint["std"]

    model.eval()

    # Normalize input
    seq = (seq - mean) / std

    with torch.no_grad():
        pred = model(seq, day, menu)

    # Denormalize output
    pred_real = pred.item() * std + mean

    return pred_real


if __name__ == "__main__":
    # Example usage
    sample = ([600, 620, 610, 605, 615, 625, 630], 1, 1, 650)  
    prediction = predict(sample)
    print(f"Predicted footfall: {prediction:.2f}")
    print(f"Actual footfall: {sample[3]:.2f}")