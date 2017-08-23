import matplotlib.pyplot as plt
import numpy as np

e1 = 0.198
s1 = 0.357
e2 = 0.055
s2 = 0.203
r = 0.18

x = np.arange(0, 1, 0.01)
e = (e1 + e2) * x + e2
v = (s1**2 - 2 * r * s1 * s2 + s2**2) * (x ** 2) - 2 * (s2**2 - r * s1 * s2) * x + s2 ** 2

plt.plot(x, e)
plt.plot(x, v)
plt.show()
