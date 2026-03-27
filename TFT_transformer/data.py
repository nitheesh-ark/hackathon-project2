import numpy as np

def generate_data(n_samples=512, seq_len=7):
    data = []

    for _ in range(n_samples):
        base = np.random.randint(200, 500)

        footfall_seq = base + np.random.randint(-50, 50, size=seq_len)

        day = np.random.randint(0, 7)
        menu = np.random.randint(0, 5)

        target = footfall_seq[-1]
        target += (15 if day in [5, 6] else -10)
        target += (20 if menu == 2 else 0)

        data.append((footfall_seq, day, menu, target))

    return data