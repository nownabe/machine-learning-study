import numpy as np

def spearman(d1, d2):
    n = d1.size
    return 1 - 6 * np.square(d1 - d2).sum() / (n ** 3 - n)

def kendall(d1, d2):
    n = d1.size
    g = 0
    h = 0
    for i in range(0, n):
        for j in range(0, n):
            if j <= i:
                continue
            d = (d1[i] - d1[j]) * (d2[i] - d2[j])
            if d > 0:
                g += 1
            elif d < 0:
                h += 1

    return (g - h) / (n * (n - 1) / 2)

d1 = np.array([
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
    21, 22, 23, 24, 25, 26, 27, 28, 29, 30
])
d2 = np.array([
    1, 5, 2, 3, 6, 7, 15, 8, 4, 11, 10,
    14, 18, 13, 22, 24, 16, 19, 30, 9, 25,
    17, 26, 23, 12, 20, 28, 21, 27, 29
])
d3 = np.array([
    8, 3, 1, 4, 2, 5, 11, 7, 15, 9,
    6, 13, 10, 22, 12, 14, 18, 19, 17, 22,
    16, 24, 21, 20, 28, 30, 25, 26, 27, 29
])
d4 = np.array([
    20, 1, 4, 2, 6, 3, 12, 17, 8, 5,
    18, 13, 23, 26, 29, 15, 16, 9, 10, 11,
    30, 7, 27, 19, 14, 21, 28, 24, 22, 25
])

data = [d1, d2, d3, d4]

scores = [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
]

for i in range(0, 4):
    for j in range(0, 4):
        if i == j:
            continue
        scores[j][i] = spearman(data[i], data[j])
        scores[i][j] = kendall(data[i], data[j])

for s in scores:
    print(s)
