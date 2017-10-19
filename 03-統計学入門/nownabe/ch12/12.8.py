import numpy as np


def kh2(data):
    x = data[0, 0]
    y = data[0, 1]
    z = data[1, 0]
    u = data[1, 1]
    n = data.sum()
    return (n * (x * u - y * z) ** 2) / ((x + z) * (y + u) * (x + y) * (z + u))


def yates(data):
    x = data[0, 0]
    y = data[0, 1]
    z = data[1, 0]
    u = data[1, 1]
    n = data.sum()
    c = - n / 2 if x * u - y * z >= 0 else n / 2
    return (n * (x * u - y * z + c) ** 2) / ((x + z) * (y + u) * (x + y) * (z + u))


data = np.array([
    [9, 12],
    [4, 5]
])

print(f"補正前: {kh2(data)}")
print(f"補正後: {yates(data)}")
