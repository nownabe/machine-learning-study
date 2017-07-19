DATA = [0, 1, 2, 3, 5, 5, 7, 8, 9, 10]

import math

class Ch2_4:
    def __init__(self, data):
        self.data = data
        self.mean_value = sum(data) / len(data)
        self.n = len(data)

        self._variance = None

    def standard_score(self, i):
        return (self.data[i] - self.mean_value) / self.standard_deviation()

    def deviation_value(self, i):
        return 10 * self.standard_score(i) + 50

    def standard_deviation(self):
        return math.sqrt(self.__variance())

    def __variance(self):
        if self._variance is None:
            val = []
            for i in self.data:
                val.append((i - self.mean_value) ** 2)

            self._variance = sum(val) / self.n

        return self._variance

model = Ch2_4(DATA)
print('S.D.')
for i in range(len(DATA)):
    print(model.standard_score(i))

print('D.V.')
for i in range(len(DATA)):
    print(model.deviation_value(i))

# or simply:

# import numpy as np
# from scipy import stats
# arr = np.array(DATA)
# stast.zscore(arr)
