# -*- coding: utf-8 -*-
import numpy as np
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt

# 自民得票率
x = np.array([
    41.4, 76.3, 59.2, 51.8, 52.5, 53.2, 62.4, 55.0, 52.7,
    63.2, 37.5, 48.5, 32.4, 20.5, 47.9, 68.9, 68.5, 52.5,
    63.3, 58.8, 59.7, 48.4, 40.7, 51.0, 50.9, 34.3, 25.8,
    32.1, 34.4, 55.1, 60.3, 57.0, 45.6, 54.2, 55.1, 55.7,
    70.3, 61.8, 47.6, 42.5, 71.3, 55.2, 65.2, 42.9, 54.7,
    62.0, 48.2,
])

# 持ち家比率
y = np.array([
    52.8, 71.2, 72.6, 63.7, 81.3, 81.8, 70.9, 74.0, 73.2,
    72.9, 66.7, 65.7, 43.7, 55.5, 79.6, 85.7, 75.3, 80.5,
    73.0, 77.0, 77.5, 69.2, 60.0, 78.2, 79.5, 61.8, 49.6,
    59.6, 72.1, 71.0, 76.3, 72.8, 71.8, 60.7, 67.0, 71.8,
    71.2, 68.3, 68.5, 54.8, 76.0, 65.8, 69.4, 66.9, 69.7,
    71.2, 59.6,
])

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x, y)
ax.set_title('Graph')
ax.set_xlabel('x')
ax.set_ylabel('y')
# fig.show()
plt.show()
