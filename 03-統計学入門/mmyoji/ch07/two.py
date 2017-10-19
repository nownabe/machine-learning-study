import numpy as np
import matplotlib.pyplot as plt

# Drawing range
x = np.arange(0, 1, 0.1)

e1 = 0.198
sig1 = 0.357
e2 = 0.055
sig2 = 0.203
rh = 0.18

## E(R) ##
e = x*e1 + (1-x)*e2

## V(R) ##
v = (sig1**2 - 2*rh*sig1*sig2 + sig2**2)*(x**2) - 2*sig2 *(sig2 - rh*sig1)*x + sig2**2

# plt.plot(x, e)
plt.plot(x, v)
plt.show()
