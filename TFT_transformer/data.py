import numpy as np

def generate_data(n_samples=512, seq_len=7):
    data = []

    for _ in range(n_samples):
        base = np.random.randint(100, 900)

        footfall_seq = base + np.random.randint(-10, 60, size=seq_len)

        day = np.random.randint(0, 7)
        menu = np.random.randint(0, 5)

        target = footfall_seq[-1]
        target += (100 if day in [5, 6] else 50)
        target += (30 if menu == 1 else 10)

        data.append((footfall_seq, day, menu, target))

    return data

if __name__ == "__main__":
    data = generate_data(n_samples=5, seq_len=7)
    for sample in data:
        print(sample)