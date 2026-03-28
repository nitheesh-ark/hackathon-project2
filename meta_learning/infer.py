import torch
from model import MetaSurplusModel
from adapt import adapt_model

class SurplusPredictor:
    def __init__(self, model_path="meta_model.pth", device="cpu"):
        self.device = device
        self.model = MetaSurplusModel().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def predict(self, x_input):
        """
        Base surplus prediction
        x_input: torch.tensor of shape [N, 4] -> [footfall_pred, food_prepared, day, menu]
        Returns: torch.tensor of predicted surplus
        """
        x_input = x_input.to(self.device)
        with torch.no_grad():
            pred = self.model(x_input)
        return pred

    def predict_adapt(self, x_input, y_real, steps=30, lr=0.001):
        """
        Adapted surplus prediction using feedback
        x_input: torch.tensor of shape [N, 4]
        y_real: torch.tensor of shape [N, 1] -> observed surplus
        Returns: torch.tensor of updated prediction
        """
        # make a copy of the model and adapt
        adapted_model = adapt_model(self.model, x_input.to(self.device), y_real.to(self.device), steps=steps, lr=lr)
        with torch.no_grad():
            new_pred = adapted_model(x_input.to(self.device))
        return new_pred



if __name__ == "__main__":

    x = torch.tensor([
        [420, 500, 2, 1],
        [380, 480, 3, 2]
    ], dtype=torch.float32)

    y_real = torch.tensor([
        [80.0],
        [120.0]
    ], dtype=torch.float32)

    s_predictor = SurplusPredictor(model_path="meta_model.pth")

    # base prediction
    base_pred = s_predictor.predict(x)
    print("Base Surplus Prediction:", base_pred)
    
    # adapted prediction
    adapted_pred = s_predictor.predict_adapt(x, y_real)
    print("Adapted Surplus Prediction:", adapted_pred)