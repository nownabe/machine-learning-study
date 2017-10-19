import random
import numpy as np
import scipy.stats as stats
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# i)
print(random.randrange(1, 12))

# ii)
X = [
    71, 68, 66, 67, 70, 71,
    70, 73, 72, 65, 66,
]
Y = [
    69, 64, 65, 63, 65, 62,
    65, 64, 66, 59, 62,
]

class Four:
    def __init__(self):
        self.arr = []
        self.x = []
        self.y = []
        self.rs = []

    def run(self):
        for i in range(200):
            val = self.__gen_colcoe()
            self.rs.append(val)

        self.__show_histogram()

    def __rand_num(self):
        return random.randrange(0, 11)

    def __init_val(self):
        self.arr = []
        self.x = []
        self.y = []

    def __gen_colcoe(self):
        self.__init_val()

        for i in range(11):
            self.arr.append(self.__rand_num())

        for i in self.arr:
            self.x.append(X[i])
            self.y.append(Y[i])

        self.x = np.array(self.x)
        self.y = np.array(self.y)
        val, _ = stats.pearsonr(self.x, self.y)
        return val

    def __show_histogram(self):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.hist(self.rs, bins=50)
        ax.set_title('histogram')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        # fig.show()
        plt.show()

f = Four()
f.run()
