import numpy as np

money = np.array([
    40000000, # 1等
    10000000,
    200000,
    10000000, # 2等
    100000,
    1000000,  # 3等
    140000,
    10000,
    1000,
    200
])

counts = np.array([7, 14, 903, 5, 645, 130, 130, 1300, 26000, 1300000])

total = 13000000

print((money * (counts / total)).sum())
