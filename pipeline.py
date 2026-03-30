from TFT_transformer.infer import predict as tft_predict
from prepare.infer import predict as meal_predict

from meta_learning.infer import predict as Surplus_predict
from redistribution.model import SmartRedistributor

day = 1
menu = 3

past_weak = [600, 620, 610, 605, 615, 625, 630]

ngos = [
    {"name": "NGO A", "distance": 2, "capacity": 10, "urgency": 5},
    {"name": "NGO B", "distance": 5, "capacity": 80, "urgency": 3},
    {"name": "NGO C", "distance": 5, "capacity": 10, "urgency": 10}
]

menu_map = {
    0: ["Pasta Alfredo", "Spaghetti Bolognese", "Margherita Pizza", "Garlic Bread"],
    1: ["Hakka Noodles", "Fried Rice", "Gobi Manchurian", "Chilli Paneer"],
    2: ["Steamed Rice", "Sambar", "Rasam", "Poriyal"],
    3: ["Butter Chicken", "Paneer Butter Masala", "Dal Tadka", "Jeera Rice"]
}

dishes = menu_map[menu]

sample = (past_weak, day, menu, 650) 
footfall_predict = tft_predict(sample, "./TFT_transformer/tft_model.pth")
print("footfall_prediction : ", footfall_predict)

prepare_meal = meal_predict(footfall_predict, day, menu, "./prepare/prep_model.pth")
for i in range(4):
    print(f"{dishes[i]} → {prepare_meal[i]:.1f} plates")
print(prepare_meal)
print()

total_meal = sum(prepare_meal)
print("total_meal : ", total_meal)


leftover_food = Surplus_predict(footfall_predict, total_meal, day, menu, "./meta_learning/meta_model_v2.pth")
for i in range(4):
    print(f"{dishes[i]} → {leftover_food[i]:.1f} plates")
print(sum(leftover_food))
print()

surplus = sum(leftover_food)

agent = SmartRedistributor()
allocation, leftover = agent.allocate(surplus, ngos)

for allocate in allocation:
    print(allocate)

print(leftover)


print("sucessuflly imported")