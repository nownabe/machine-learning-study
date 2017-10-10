import numpy as np

freq = np.array([
    [950, 348],
    [117, 54]
])

n = freq.sum()
print(f"n: {n}")

kh2 = 0
fi = freq.sum(axis=1)
fj = freq.sum(axis=0)

print(fi[0])
print(fi[1])
print(fj[0])
print(fj[1])

for i in range(2):
    for j in range(2):
        kh2 += ((n * freq[i, j] - fi[i] * fj[j]) ** 2) / (n * fi[i] * fj[j])

print(f"検定統計量: {kh2}")

ddof = (freq.shape[0] - 1) * (freq.shape[1] - 1)
print(f"自由度: {ddof}")

a = 0.05
print(f"有意水準: {a}")

kha = 3.84146
print(f"上側確率{a} パーセント点: {kha}")

print()
if kh2 > kha:
    print("関連がある (独立ではない)")
else:
    print("関連はない (独立である)")
