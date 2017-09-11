import matplotlib.pyplot as plt
import numpy as np


def gaussian(mu, sigma2):
    def g(x):
        return (1 / np.sqrt(2 * np.pi * sigma2)) * np.exp(- (x - mu) ** 2 / (2 * sigma2))
    return g


def simple_random_work(n, p):
    return gaussian(n * (2 * p - 1), 4 * n * p * (1 - p))


def srw_range(n, p, m):
    mu = n * (2 * p - 1)
    sigma2 = 4 * n * p * (1 - p)
    min = mu - m * np.sqrt(sigma2)
    max = mu + m * np.sqrt(sigma2)
    return np.arange(min, max, 0.01)

x = srw_range(20, 0.4, 4)

s10 = simple_random_work(10, 0.4)
s20 = simple_random_work(20, 0.4)

plt.plot(x, s10(x), label="S10")
plt.plot(x, s20(x), label="S20")

plt.legend()
plt.show()
