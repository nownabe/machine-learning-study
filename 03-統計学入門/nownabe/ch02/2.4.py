import numpy as np

def zscore(data):
    return (data - data.mean()) / data.std()

def tscore(data):
    return 10 * zscore(data) + 50

B = np.array([0, 1, 2, 3, 5, 5, 7, 8, 9, 10])

print("標準得点: ", zscore(B))
print("偏差値得点: ", tscore(B))
