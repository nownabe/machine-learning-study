import numpy as np

n = 50
p = 27 / 50
z = 1.96

lower = p - z * np.sqrt(p * (1 - p) / n)
upper = p + z * np.sqrt(p * (1 - p) / n)

print(f"n: {n}")
print(f"p: {p}")
print(f"a: 0.05")
print(f"zの上側確率2.5%点: {z}")

print("(11.59)式より")
print("信頼区間: [{0:0.4f}, {1:0.4f}]".format(lower, upper))
