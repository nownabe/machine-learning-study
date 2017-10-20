import numpy as np
import scipy.stats as stats

# (1)
x1 = np.array([
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
    21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
])

# (2)
x2 = np.array([
    1, 5, 2, 3, 6, 7, 15, 8, 4, 11,
    10, 14, 18, 13, 22, 24, 16, 19, 30, 9,
    25, 17, 26, 23, 12, 20, 28, 21, 27, 29,
])

print(stats.spearmanr(x1, x2))
print(stats.kendalltau(x1, x2))
# SpearmanrResult(correlation=0.82157953281423801, pvalue=2.6278827383838321e-08)
# KendalltauResult(correlation=0.66436781609195406, pvalue=2.5220285989113868e-07)
