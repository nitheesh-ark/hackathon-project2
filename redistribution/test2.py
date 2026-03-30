import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



class simple_nn(nn.Module):
    def __init__(self):
        super(simple_nn,self).__init__()

        self.fc1 = nn.Linear(3, 9)
        self.fc2 = nn.Linear(9, 3)
        self.fc3 = nn.Linear(3, 1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        
        return x






class SmartRedistributor:
    def __init__(self):

        self.model = simple_nn()

    def compute_score(self, ngo):

        score = self.model.forward(torch.tensor([
            ngo["distance"],
            ngo["capacity"],
            ngo["urgency"]
        ], dtype=torch.float32))
        print("Computed score for {}: {:.4f}".format(ngo["name"], score.item()))
        return score.item()

    def allocate(self, surplus, ngos):

        for ngo in ngos:
            ngo["score"] = self.compute_score(ngo)


        ngos = sorted(ngos, key=lambda x: x["score"], reverse=True)

        allocation = []
        remaining = surplus

        for ngo in ngos:
            if remaining <= 0:
                break

            give = min(ngo["capacity"], remaining)

            allocation.append({
                "ngo": ngo["name"],
                "allocated": int(give),
                "distance": ngo["distance"]
            })

            remaining -= give

        return allocation, remaining






model = SmartRedistributor()
surplus = 10+80+49

ngos = [
    {"name": "NGO A", "distance": 2, "capacity": 50, "urgency": 5},
    {"name": "NGO B", "distance": 5, "capacity": 80, "urgency": 5},
    {"name": "NGO C", "distance": 5, "capacity": 10, "urgency": 10}
]

allocation, lefts = model.allocate(surplus, ngos)

print("allactiion plan")
for a in allocation:
    print(a)

print("left over foods :::", lefts)

