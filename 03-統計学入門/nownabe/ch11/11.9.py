import numpy as np

x = np.array([4, 3, 5, 4, 8, 2, 5, 9, 3, 5])

n = x.size
a = 0.01
z = 2.58 # 付表1より
l = x.mean()

lower = l - z * np.sqrt(l / n)
upper = l + z * np.sqrt(l / n)

print(f"X: {x}")
print(f"n: {n}")
print(f"a: {a}")
print(f"Zの上側確率0.5%点: {z}")
print(f"lambda: {l}")

print("\n230ページの式より")
print("信頼区間: [{0:0.4f}, {1:0.4f}]".format(lower, upper))
