import numpy as np
import torch

def generate_dataset(n_samples=1000):
    data = []
    targets = []

    for _ in range(n_samples):
        distance = np.random.uniform(0, 10)   
        capacity = np.random.randint(10, 100)
        urgency = np.random.randint(1, 6)     


        score = 0

        # urgency rule
        if urgency >= 4:
            score += 0.5

        # distance rule
        if distance < 5:
            score += 0.3

        # capacity rule
        if capacity > 50:
            score += 0.2

        # add noise (real-world randomness)
        score += np.random.normal(0, 0.05)

        # clamp between 0 and 1
        score = max(0, min(1, score))

        data.append([distance, capacity, urgency])
        targets.append([score])

    X = torch.tensor(data, dtype=torch.float32)
    y = torch.tensor(targets, dtype=torch.float32)

    return X, y