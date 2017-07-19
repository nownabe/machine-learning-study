DATA = {
    'this year':    [32, 19, 10, 24, 15],
    '10 years ago': [28, 13, 18, 29, 12],
}

from math import log10

class Ch2_3:
    def __init__(self, data):
        self.data = data
        self.n = sum(data)

    def entropy(self):
        values = []
        for i in self.data:
            v = i / self.n
            values.append(v * log10(v))

        return sum(values) * -1

for k, v in DATA.items():
    model = Ch2_3(v)

    print(k + ': ' + str(model.entropy()))
# this year:    0.667724435887455
# 10 years ago: 0.6704368955892825
