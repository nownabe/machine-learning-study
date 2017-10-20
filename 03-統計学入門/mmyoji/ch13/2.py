import numpy as np
import matplotlib.pyplot as plt
import math

# 銅消費量(千t)
x = np.array([
    229, 367, 301, 352, 457, 427, 485, 616,
    695, 806, 815, 826, 951, 1202, 881, 827, 1050, 1127, 1241,
    1330, 1158, 1254, 1243, 1216, 1368, 1231, 1219, 1284, 1355,
])

# 実質GDP(兆円)
y = np.array([
    61.2, 70.0, 74.9, 82.8, 93.6, 98.5, 108.8, 120.1,
    135.1, 152.5, 165.8, 173.0, 189.9, 202.6, 199.7, 205.0, 214.9, 226.3, 238.1,
    250.7, 261.4, 271.0, 279.3, 288.4, 303.0, 317.3, 325.7, 340.3, 359.5,
])

n = len(x)
print("n:", n)

## i.
# plt.scatter(x, y)
# plt.show()

## ii.
x_mean = x.mean()
y_mean = y.mean()
print("X mean:", x_mean)
print("Y mean:", y_mean)
# X mean: 907.344827586
# Y mean: 200.324137931

b2_hat = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
b1_hat = y_mean - b2_hat * x_mean
print("^B2:", b2_hat)
print("^B1:", b1_hat)
# ^B2: 0.229766115486
# ^B1: -8.15295850996

print('Y = {b1_hat} + {b2_hat}X'.format(b1_hat=b1_hat, b2_hat=b2_hat))
# Y = -8.15295850995733 + 0.22976611548621453X

# ???

## iii.
