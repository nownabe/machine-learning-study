import numpy as np
import math

# Men's salary
data1 = np.array([
    15.4, 18.3, 16.5, 17.4, 18.9,
    17.2, 15.0, 15.7, 17.9, 16.5,
])
m = 10

# Women's salary
data2 = np.array([
    14.2, 15.9, 16.0, 14.0, 17.0,
    13.8, 15.2, 14.5, 15.0, 14.4,
])
n = 10

## i) ##
#
# null hypothesis
#   mu1 == mu2
#
# alternative hypothesis
#   mu1 != mu2
#
# use 12.8, 12.9

alpha = 0.05

x = np.mean(data1)
s1 = np.var(data1, ddof=1)
y = np.mean(data2)
s2 = np.var(data2, ddof=1)

# 12.8
s_2 = ((m-1) * s1 + (n-1) + s2) / (m + n - 2)
s = math.sqrt(s_2)

# t_{0.025}(18) = 2.101
t1 = 2.101

# 12.9
t = (x - y) / (s * math.sqrt((1/m) + (1/n)))
print("accept" if abs(t) <= t1 else "reject")
# "reject"

## ii) ##
#
# Welch's test
#
# use 12.10, 12.11

t = (x - y) / math.sqrt((s1/m) + (s2/n))
v = round(((s1/m) + (s2/n))**2 / ((((s1/m) ** 2) / (m-1)) + (((s2/n) ** 2) / (n-1))))
print("v =", v)
# v = 17.0

# t_{0.025}(17) = 2.110
t1 = 2.110
print("accept" if abs(t) <= t1 else "reject")
# "reject"

## iii) ###
# use 12.12

alpha = 0.01

f = s1 / s2
print("F =", f)
# F = 1.56352201258

# alpha / 2 = 0.005
# F_{0.995}(9, 9) = ???
# F_{0.005}(9, 9) = 6.541

f1 = 6.541
print("accept" if f <= f1 else "reject")
# "accept"
