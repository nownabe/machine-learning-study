import numpy as np


def pooled_variance(x, y):
    return (((x - x.mean()) ** 2).sum() + ((y - y.mean()) ** 2).sum()) / (x.size + y.size - 2)

x  = np.array([7.97, 7.66, 7.59, 8.44, 8.05, 8.08, 8.35, 7.77, 7.98, 8.15])
y  = np.array([8.06, 8.27, 8.45, 8.05, 8.51, 8.14, 8.09, 8.15, 8.16, 8.42])
s2 = pooled_variance(x, y)
s  = np.sqrt(s2)
t  = 2.101

lower = x.mean() - y.mean() - t * s * np.sqrt(1 / x.size + 1 / y.size)
upper = x.mean() - y.mean() + t * s * np.sqrt(1 / x.size + 1 / y.size)

print(f"X: {x}")
print(f"Y: {y}")

print("Pooled variance: {0:0.4f}".format(s2))
print("s: {0:0.4f}".format(s))
print("Mean of X: {0:0.4f}".format(x.mean()))
print("Mean of Y: {0:0.4f}".format(y.mean()))

print("t = {0:0.3f}".format(t))

print("[{0:0.4f}, {1:0.4f}]".format(lower, upper))
