import numpy as np

def entropy(data):
    p = data / data.sum()
    return -np.sum(p * np.log10(p))

d1 = np.array([32, 19, 10, 24, 15])
d2 = np.array([28, 13, 18, 29, 12])

print("本年: %0.3f" % entropy(d1))
print("10年前: %0.3f" % entropy(d2))
