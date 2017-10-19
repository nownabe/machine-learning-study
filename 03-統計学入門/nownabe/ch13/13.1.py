import numpy as np
import matplotlib.pyplot as plt

x = np.array([
    2, 2, 2, 2, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5,
    3, 3, 3, 3, 3, 3, 3, 3.5, 3.5, 3.5,
    3.5, 3.5, 3.5, 4, 4, 4, 4, 4, 4, 4,
    4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 5, 5, 5, 5,
    5, 5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 6, 6,
    6, 6, 6, 6.5, 6.5, 6.5, 6.5, 7
])

y = np.array([
    2, 2.5, 2.5, 3, 2, 2.5, 3, 3, 3, 3.5,
    2.5, 3, 3, 3.5, 3.5, 4, 4.5, 3, 3.5, 4,
    4.5, 5, 5.5, 3.5, 4, 4.5, 4.5, 5, 5.5, 5.5,
    4, 4.5, 5, 5, 5.5, 5.5, 6, 4.5, 5, 5.5,
    6, 6.5, 5, 5.5, 5.5, 6, 6.5, 7, 5.5, 5.5,
    6, 6.5, 7, 5.5, 6.5, 7, 7, 7.5
])

print(f"直径: {x}")
print(f"樹高: {y}")

print("\n(1) 散布図を作成せよ")
#plt.scatter(x, y)
#plt.show()

print("\n(2) 樹高を胸高直径へ回帰せよ")

b2 = ((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum()
b1 = y.mean() - b2 * x.mean()

print("Y = {0:0.4f} + {1:0.4f}X".format(b1, b2))

print("\n(3) b2 = 0.9 を有意水準5%で仮説検定せよ")

print("帰無仮説 H0: b2 == 0.9")
print("対立仮説 H1: b2 != 0.9")

n = x.size
print(f"n = {n}")

a = 0.9

se = np.sqrt(((y - b1 - b2 * x) ** 2).sum() / (n - 2))
seb2 = se / np.sqrt(((x - x.mean()) ** 2).sum())
t2 = (b2 - a) / seb2
print(f"t2 = {t2}")

# 付表2より
alpha_0025 = 2.009 + (2.000 - 2.009) / (60 - 50) * (56 - 50)
print(f"alpha_0.025 = {alpha_0025}")

if np.abs(t2) > alpha_0025:
    print("帰無仮説を棄却する。")
else:
    print("帰無仮説は棄却されない。")

print("\n(3) 標本回帰直線から2s.e.以上はずれる樹木はあるか")

hy = b1 + b2 * x
check = (y > hy + 2 * se) | (y < hy - 2 * se)
if check.any():
    print("ある")
    print(np.array([x[check], y[check]]).T)
else:
    print("ない")

print("\n(4) 胸高直径8寸の樹高の平均値を推定せよ")
print(f"{b1 + b2 * 8}尺")
