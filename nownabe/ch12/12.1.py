import numpy as np

x = np.array([101.1, 103.2, 102.1, 99.2, 100.5, 101.3, 99.7, 100.5, 98.9, 101.4])
print(f"測定値: {x}")

mu = 100
print(f"帰無仮説 H0: u0 == {mu}")
print(f"対立仮説 H1: u0 != {mu}")

a = 0.05
print(f"有意水準: {a}")

s = x.std(ddof=1)
t = (x.mean() - mu) / (s / np.sqrt(x.size))
print(f"検定統計量 t: {t}")

ta = 2.262
print(f"上側確率{a/2} パーセント点: {ta}")

print()
if np.abs(t) > ta:
    print("H0は棄却される")
else:
    print("H0は棄却されない")
