import numpy as np

freq = np.array([10, 7, 8, 11, 6, 8])
n = freq.sum()

kh2 = ((freq - n / 6) ** 2).sum() / (n / 6)
print(f"検定統計量: {kh2}")

a = 0.05
print(f"有意水準: {a}")

kha = 11.0705
print(f"上側確率{a} パーセント点: {kha}")

print()
if kh2 > kha:
    print("さいころは正しく作られていない")
else:
    print("さいころは正しく作られている")
