import numpy as np
import matplotlib.pyplot as plt
import math

x = np.array([
    2, 2, 2, 2, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 3, 3, 3, 3, 3, 3, 3, 3.5,
    3.5, 3.5, 3.5, 3.5, 3.5, 4, 4, 4, 4, 4, 4, 4, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 5, 5,
    5, 5, 5, 5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 6, 6, 6, 6, 6, 6.5, 6.5, 6.5, 6.5, 7,
])

y = np.array([
    2, 2.5, 2.5, 3, 2, 2.5, 3, 3, 3, 3.5, 2.5, 3, 3, 3.5, 3.5, 4, 4.5, 3,
    3.5, 4, 4.5, 5, 5.5, 3.5, 4, 4.5, 4.5, 5, 5.5, 5.5, 4, 4.5, 5, 5, 5.5, 5.5, 6, 4.5,
    5, 5.5, 6, 6.5, 5, 5.5, 5.5, 6, 6.5, 7, 5.5, 5.5, 6, 6.5, 7, 5.5, 6.5, 7, 7, 7.5,
])

n = len(x)
print("n:", n)
# n: 58

## i.

# plt.scatter(x, y)
# plt.show()

## ii.

# 13.5
x_mean = x.mean()
y_mean = y.mean()
print("X mean:", x_mean)
print("Y mean:", y_mean)
# X mean: 4.24137931034
# Y mean: 4.68103448276

b2_hat = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
b1_hat = y_mean - b2_hat * x_mean
print("^B2:", b2_hat)
print("^B1:", b1_hat)
# ^B2: 0.932447310475
# ^B1: 0.726171752123

print('Y = {b1_hat} + {b2_hat}X'.format(b1_hat=b1_hat, b2_hat=b2_hat))
# Y = 0.7261717521233093 + 0.9324473104749922X


## iii.
#
# H0: B2 = 0.9
# H1: B2 != 0.9
#
a = 0.9
# alpha = 0.05

# 13.13
# s.e.
se = math.sqrt(
    np.sum((y - b1_hat - b2_hat * x) ** 2) / (n - 2)
)
print("s.e.:", se)
# s.e.: 0.6619580504417524

# 13.14
# s.e.(^B2)
se_b2_hat = se / math.sqrt(np.sum((x - x_mean) ** 2))
print("s.e.(^B2):", se_b2_hat)
# s.e.(^B2): 0.06322432526628806

# 13.15
t2 = (b2_hat - a) / se_b2_hat
print("t2:", t2)
# t2: 0.513209280421
# t{0.025}(56) # https://staff.aist.go.jp/t.ihara/tinv.html
t = 2.0032
print("reject" if abs(t2) >= t else "approve")
# approve
# => 仮説はあっている

## iv
print((y - b1_hat - b2_hat * x) > 2 * se)
# 1 True
# 2s.e. を外れる樹木はある

## v.
print(b2_hat * 8.0 + b1_hat)
# 8.18575023592
