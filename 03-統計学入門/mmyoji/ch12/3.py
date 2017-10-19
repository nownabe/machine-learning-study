import numpy as np
import math

# 高血圧治療なので、治療後の血圧が下がっていれば OK
# => 片側検定 (P.241)
#
# H0: \mu = 0
# H1: \mu < 0

alpha = 0.01
data = np.array([
    2.0, -5.0, -4.0, -8.0, 3.0, 0.0,
    3.0, -6.0, -2.0, 1.0, 0.0, -4.0,
])
n = 12

x = np.mean(data)
s = math.sqrt(np.var(data, ddof=1))
t = (x - 0) / (s / math.sqrt(n))
print("x =", x)
print("s =", s)
print("t =", t)
# x = -1.66666666667
# s = 3.7009417310962487
# t = -1.56000907644

# t_{0.01}(11)
t1 = 2.718

print("approve" if t <= t1 else "reject")
# "approve"
