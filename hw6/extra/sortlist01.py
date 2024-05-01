import numpy as np

d0 = [
    {'f1': 0.9, 'p': 0.8, 'r': 0.78},
    {'f1': 0.85, 'p': 0.9, 'r': 0.8},
    {'f1': 0.95, 'p': 0.9, 'r': 0.75},
]

sorted_list = sorted(d0, key=lambda x: x['f1'], reverse=True)
print(sorted_list)

best_d = sorted_list[0]
print(best_d['f1'])