marks = [
    {"maths":8, "science":9, "value":8, "no" : 1},
    {"maths":10, "science":10, "value":10,"no" : 2},
    {"maths":10, "science":9, "value":10, "no" : 3}
]
print(marks)
def score(mark):
    return mark["maths"] + mark["science"] + mark["value"]

for mark in marks:
    mark["score"] = score(mark)

for mark in marks:
    print(mark["no"], mark["score"])

sorteds = sorted(marks, key = lambda x: x["score"], reverse=True)
print()


for mark in sorteds:
    print(mark["no"], mark["score"])
