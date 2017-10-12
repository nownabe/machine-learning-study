import numpy as np
import math

n = 10
data = np.array([
    101.1, 103.2, 102.1, 99.2, 100.5,
    101.3, 99.7, 100.5, 98.9, 101.4,
])
mean = np.mean(data)
var  = np.var(data, ddof=1)
print("mean =", mean)
print("var  =", var)
# mean = 100.79
# var  = 1.74544444444

s = math.sqrt(var)
print("s =", s)
# s = 1.3211526953552493

# t_{0.025}(9) = 2.262
t1 = 2.262

t = (mean - 100) / (s / math.sqrt(10))
print("t =", t)
# t = 1.89092400925
print("accept" if abs(t) <= t1 else "reject")
# "accept"
