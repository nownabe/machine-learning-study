import numpy as np

x = np.array([18, 84])
y = np.array([8, 93])

print("帰無仮説 H0: p1 == p2")
print("対立仮説 H1: p1 != p2")

p1 = x[0] / x.sum()
p2 = y[0] / y.sum()
p = (x[0] + y[0]) / (x.sum() + y.sum())

print(f"p1: {p1}")
print(f"p2: {p2}")
print(f"p: {p}")

z = (p1 - p2) / np.sqrt((1 / x.sum() + 1 / y.sum()) * p * (1 - p))
print(f"検定統計量 z: {z}")

a = 0.05
print(f"有意水準: {a}")

za = 1.960
print(f"上側確率{a/2} パーセント点: {za}")

print()
if np.abs(z) > za:
    print("帰無仮説は棄却される")
else:
    print("帰無仮説は棄却されない")
