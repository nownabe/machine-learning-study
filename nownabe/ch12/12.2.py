import numpy as np

x = np.array([15.4, 18.3, 16.5, 17.4, 18.9, 17.2, 15.0, 15.7, 17.9, 16.5])
y = np.array([14.2, 15.9, 16.0, 14.0, 17.0, 13.8, 15.2, 14.5, 15.0, 14.4])

print(f"男: {x}")
print(f"女: {y}")

print()
print("(1)")

print("帰無仮説 H0: 男の平均賃金 == 女の平均賃金")
print("対立仮説 H1: 男の平均賃金 != 女の平均賃金")

a = 0.05
print(f"有意水準: {a}")

s2 = ((x.size - 1) * x.var(ddof=1) + (y.size - 1) * y.var(ddof=1)) / (x.size + y.size - 2)
t = (x.mean() - y.mean()) / (np.sqrt(s2) * np.sqrt(1 / x.size + 1 / y.size))
print(f"検定統計量 t: {t}")

ta = 2.101
print(f"上側確率{a/2} パーセント点: {ta}")

print()
if np.abs(t) > ta:
    print("H0は棄却される")
else:
    print("H0は棄却されない")

print()
print("(2)")

t = (x.mean() - y.mean()) / np.sqrt(x.var(ddof=1) / x.size + y.var(ddof=1) / y.size)
print(f"検定統計量 t: {t}")

sx2 = x.var(ddof=1)
sy2 = y.var(ddof=1)
numer = (sx2 / x.size + sy2 / y.size) ** 2
denom = (sx2 / x.size) ** 2 / (x.size - 1) + (sy2 / y.size) ** 2 / (y.size - 1)
n = int(np.round(numer / denom))
print(f"n: {n}")

ta = 2.110
print(f"上側確率{a/2} パーセント点: {ta}")

print()
if np.abs(t) > ta:
    print("H0は棄却される")
else:
    print("H0は棄却されない")

print()
print("(3)")

print("帰無仮説 H0: 男の賃金分散 == 女の賃金分散")
print("対立仮説 H1: 男の賃金分散 != 女の賃金分散")

a = 0.01
print(f"有意水準: {a}")

f = sx2 / sy2
print(f"検定統計量 f: {f}")

fau = 6.541
fal = 1 / fau
print(f"上側確率{a/2} パーセント点: {fau}")
print(f"上側確率{1-a/2} パーセント点: {fal}")

if f < fal or f > fau:
    print("H0は棄却される")
else:
    print("H0は棄却されない")
