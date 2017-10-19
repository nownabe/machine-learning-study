import numpy as np

data = np.array([1.22, 1.24, 1.25, 1.19, 1.17, 1.18])

sigma2 = ((data - data.mean()) ** 2).sum() / (data.size - 1)

print("標本平均: {0:0.2f}".format(data.mean()))
print("標本分散: {0:0.6f}".format(sigma2))
