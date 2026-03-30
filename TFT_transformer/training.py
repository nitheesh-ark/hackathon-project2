import torch
import torch.nn as nn
from dataloader import FoodDataset, DataLoader
from model import TFTModel
from data import generate_data

# nessary things needed for the modeltraining
model = TFTModel()
checkpoint = torch.load("tft_model.pth", weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"])
print("checkpoint loaded successfully")

optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
loss_fn = nn.MSELoss()

#data segementation and data pipeline loading 
data = generate_data(n_samples=32, seq_len=7)
dataset = FoodDataset(data)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)



for epoch in range(1000):
    total_loss = 0

    for x_seq, day, menu, target in train_loader:
        pred = model(x_seq, day, menu).squeeze()

        loss = loss_fn(pred, target)

        optimizer.zero_grad()
        loss.backward()
       
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")


torch.save({
    "model_state_dict": model.state_dict(),
    "mean": dataset.mean,
    "std": dataset.std
}, "tft_model.pth")


# from hear  its in inference mod
model.eval()
seq, day, menu, target = dataset[0]

seq = seq.unsqueeze(0)
day = day.unsqueeze(0)
menu = menu.unsqueeze(0)

with torch.no_grad():
    pred = model(seq, day, menu)

# from normalize values to actual values
# this conversion takes place only on inference time 
pred_real = pred.item() * dataset.std + dataset.mean
target_real = target.item() * dataset.std + dataset.mean

print("Actual:", target_real)
print("Predicted:", pred_real)