import torch
from model import MetaSurplusModel

def adapt_model(base_model, x_new, y_new, steps=1, lr=0.001):
    model = MetaSurplusModel()
    model.load_state_dict(base_model.state_dict())

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    for _ in range(steps):
        optimizer.zero_grad()
        preds = model(x_new)
        loss = loss_fn(preds, y_new)
        loss.backward()
        optimizer.step()

    return model