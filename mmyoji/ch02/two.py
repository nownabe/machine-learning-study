DATA = {
    'A': [0, 3, 3, 5, 5, 5, 5, 7, 7, 10],
    'B': [0, 1, 2, 3, 5, 5, 7, 8, 9, 10],
    'C': [3, 4, 4, 5, 5, 5, 5, 6, 6, 7],
}

class Ch2_2:
    def __init__(self, data):
        self.data = data
        self.mean = sum(data) / len(data)
        self.n = len(data)

    def mean_diff(self):
        result = []

        for i in self.data:
            for j in self.data:
                result.append((abs(i - j) / (self.n ** 2)))

        return sum(result)

    def gini_coefficient(self):
        result = []

        for i in self.data:
            for j in self.data:
                result.append((abs(i-j) / (2 * self.n ** 2 * self.mean)))

        return sum(result)

for k, v in DATA.items():
    model = Ch2_2(v)

    print(k + ':')
    print('Mean diff: ' + str(model.mean_diff()))
    print('Gini:      ' + str(model.gini_coefficient()))
# A:
#    Mean diff: 2.76
#    Gini:      0.2760000000000002
# B:
#    Mean diff: 3.7599999999999976
#    Gini:      0.3760000000000002
# C:
#    Mean diff: 1.2000000000000008
#    Gini:      0.12000000000000008
