from torch.utils.data import Dataset, DataLoader
from data import generate_data
import torch
import numpy as np
class FoodDataset(Dataset):
    def __init__(self, data):
        self.data = data

        # compute normalization stats
        all_values = [x for seq, _, _, t in data for x in seq] + [t for _, _, _, t in data]
        self.mean = sum(all_values) / len(all_values)
        self.std = (sum((x - self.mean) ** 2 for x in all_values) / len(all_values)) ** 0.5

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq, day, menu, target = self.data[idx]

        seq = (torch.tensor(seq, dtype=torch.float32) - self.mean) / self.std
        target = (torch.tensor(target, dtype=torch.float32) - self.mean) / self.std

        return (
            seq.unsqueeze(-1),
            torch.tensor(day, dtype=torch.long),
            torch.tensor(menu, dtype=torch.long),
            target
        )
data = generate_data(n_samples=1000, seq_len=7)

# Create dataset
dataset = FoodDataset(data)

# Create DataLoader
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

if __name__ == "__main__":
    for x_seq, day, menu, target in train_loader:
        print("Sequence shape:", x_seq.shape)  # (batch_size, seq_len, 1)
        print("Day shape:", day.shape)          # (batch_size,)
        print("Menu shape:", menu.shape)        # (batch_size,)
        print("Target shape:", target.shape)    # (batch_size,)
        break
    print(dataset.len())