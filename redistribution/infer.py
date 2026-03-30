from model import SmartRedistributor
surplus = 100
ngos = [
    {"name": "NGO A", "distance": 2, "capacity": 10, "urgency": 5},
    {"name": "NGO B", "distance": 5, "capacity": 80, "urgency": 3},
    {"name": "NGO C", "distance": 5, "capacity": 10, "urgency": 10}
]

agent = SmartRedistributor()
allocation, leftover = agent.allocate(surplus, ngos)

print("Allocation Plan v")
for a in allocation:
    print(a)

print("Remaining Waste ::: ", leftover)