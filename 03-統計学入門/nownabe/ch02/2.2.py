import numpy as np

def mean_difference(data):
    return np.abs(np.subtract.outer(data, data)).mean()

def gini_coefficient(data):
    return mean_difference(data) / (data.mean() * 2.0)

A = np.array([0, 3, 3, 4, 4, 4, 4, 7, 7, 10])
B = np.array([0, 1, 2, 3, 5, 5, 7, 8, 9, 10])
C = np.array([3, 4, 4, 5, 5, 5, 5, 6, 6, 7])

print("平均差")
print("A: %f" % mean_difference(A))
print("B: %f" % mean_difference(B))
print("C: %f" % mean_difference(C))

print("ジニ係数")
print("A: %f" % gini_coefficient(A))
print("B: %f" % gini_coefficient(B))
print("C: %f" % gini_coefficient(C))
