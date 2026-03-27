model.eval()

sample = dataset[0]

seq, day, menu, target = sample

seq = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
day = torch.tensor([day])
menu = torch.tensor([menu])

pred = model(seq, day, menu)

print("Actual:", target)
print("Predicted:", pred.item())