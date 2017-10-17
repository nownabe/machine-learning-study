import numpy as np
import matplotlib.pyplot as plt

years = np.arange(1960, 1988 + 1)

y = np.array([
    229, 367, 301, 352, 457, 427, 485, 616,
    695, 806, 815, 826, 951, 1202, 881, 827, 1050, 1127, 1241,
    1330, 1158, 1254, 1243, 1216, 1368, 1231, 1219, 1284, 1355
])

x = np.array([
    61.2, 70.0, 74.9, 82.8, 93.6, 98.5, 108.8, 120.1,
    135.1, 152.5, 165.8, 173.0, 189.9, 202.6, 199.7, 205.0, 214.9, 226.3, 238.1,
    250.7, 261.4, 271.0, 279.3, 288.4, 303.0, 317.3, 325.7, 340.3, 359.5
])

print(f"年: {years}")
print(f"銅消費量: {x}")
print(f"実質GDP: {y}")

print("\n(1) 銅消費量とGDPをグラフに書いてその関係を分析せよ")

fig = plt.figure()

ax1 = fig.add_subplot(211)
ax1.scatter(x, y)

ax2 = fig.add_subplot(212)
ax2.plot(years, x)

ax3 = ax2.twinx()
ax3.plot(years, y)

# plt.show()

print("\n(2) 銅消費量のGDP弾性値を回帰分析によって推定せよ")

logx = np.log(x)
logy = np.log(y)

b2 = ((logx - logx.mean()) * (logy - logy.mean())).sum() / ((logx - logx.mean()) ** 2).sum()
b1 = logy.mean() - b2 * logx.mean()

print("log Y = {0:0.4f} + {1:0.4f} * log X".format(b1, b2))

print("\n(3) 実質GDPが4%/年で増加すると仮定して2000年の銅消費量を推定せよ")
print(1355 * ((1 + (4 * b2 * 0.01)) ** (2000 - 1988)))

print("\n(4) 銅消費量のGDP弾性値が1であるという仮説を5%有意水準で検定せよ")

print("帰無仮説 H0: b2 == 1")
print("対立仮説 H1: b2 != 1")

n = x.size
print(f"n = {n}")

a = 1

se = np.sqrt(((logy - b1 - b2 * logx) ** 2).sum() / (n - 2))
seb2 = se / np.sqrt(((logx - logx.mean()) ** 2).sum())
t2 = (b2 - a) / seb2
print(f"t2 = {t2}")

# 付表2より
alpha_0025 = 2.052

if np.abs(t2) > alpha_0025:
    print("帰無仮説を棄却する。")
else:
    print("帰無仮説は棄却されない。")
