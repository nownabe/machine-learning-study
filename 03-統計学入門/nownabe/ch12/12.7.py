import numpy as np

freq = np.array([
    [36, 67, 49],
    [31, 60, 49],
    [58, 87, 80],
])

n = freq.sum()
print(f"n: {n}")

kh2 = 0
fi = freq.sum(axis=1)
fj = freq.sum(axis=0)

for i in range(freq.shape[0]):
    for j in range(freq.shape[1]):
        kh2 += ((n * freq[i, j] - fi[i] * fj[j]) ** 2) / (n * fi[i] * fj[j])

print(f"検定統計量: {kh2}")

ddof = (freq.shape[0] - 1) * (freq.shape[1] - 1)
print(f"自由度: {ddof}")

a = 0.05
print(f"有意水準: {a}")

kha = 9.48773
print(f"上側確率{a} パーセント点: {kha}")

print()
if kh2 > kha:
    print("関連がある (独立ではない)")
else:
    print("関連はない (独立である)")
