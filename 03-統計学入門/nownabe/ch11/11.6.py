import numpy as np


def sample_variance(x):
    return ((x - x.mean()) ** 2).sum() / (x.size - 1)

def nu(x, y):
    x_s2 = sample_variance(x)
    y_s2 = sample_variance(y)

    numerator = (x_s2 / x.size + y_s2 / y.size) ** 2
    denominator = (x_s2 / x.size) ** 2 / (x.size - 1) + (y_s2 / y.size) ** 2 / (y.size - 1)

    return int(np.round(numerator / denominator))


x = np.array([25, 24, 25, 26])
y = np.array([23, 18, 22, 28, 17, 25, 19, 16])
a = 0.05

x_s2 = sample_variance(x)
y_s2 = sample_variance(y)
n = nu(x, y) #=> 8
t = 2.306 # 付表2より

lower = x.mean() - y.mean() - t * np.sqrt(x_s2 / x.size + y_s2 / y.size)
upper = x.mean() - y.mean() + t * np.sqrt(x_s2 / x.size + y_s2 / y.size)

print(f"X: {x}")
print(f"Y: {y}")
print(f"a: {a}")

print("Mean of X: {0:0.4f}".format(x.mean()))
print("Mean of Y: {0:0.4f}".format(y.mean()))
print("Variance of X: {0:0.4f}".format(x_s2))
print("Variance of Y: {0:0.4f}".format(y_s2))

print(f"nu: {n}")
print("t = {0:0.3f}".format(t))

print("[{0:0.4f}, {1:0.4f}]".format(lower, upper))
