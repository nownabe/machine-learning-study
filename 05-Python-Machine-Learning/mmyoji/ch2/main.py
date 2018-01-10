import pandas as pd

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
df.tail()

import matplotlib.pyplot as plt
import numpy as np

# Extract 1~100 rows' objective-values
y = df.iloc[0:100, 4].values

# Convert
#  'Iris-setosa'    as -1
#  'Iris-virginica' as 1
y = np.where(y == 'Iris-setosa', -1, 1)

# Extract 1 and 3 columns from 1~100 rows
X = df.iloc[0:100, [0, 2]].values

# plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
# plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
#
# plt.xlabel('sepal length [cm]')
# plt.ylabel('petal length [cm]')
# plt.legend(loc='uppper left')
# plt.show()

# import perceptron

# ppn = perceptron.Perceptron(eta=0.1, n_iter=10)
# ppn.fit(X, y)

# plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
# plt.xlabel('Epochs')
# plt.ylabel('# of misclassifications')
# plt.show()

from my_functions import plot_decision_regions

# plot_decision_regions(X, y, classifier=ppn)

# plt.xlabel('sepal length [cm]')
# plt.ylabel('petal length [cm]')
# plt.legend(loc='uppper left')
# plt.show()

from adaline_gd import AdalineGD

# fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
# ada1 = AdalineGD(n_iter=10, eta=0.01).fit(X, y)
# ax[0].plot(range(1, len(ada1.cost_)+1), np.log10(ada1.cost_), marker='o')
# ax[0].set_xlabel('Epochs')
# ax[0].set_ylabel('log(Sum-squared-error)')
# ax[0].set_title('Adaline - Learning rate 0.01')

# ada2 = AdalineGD(n_iter=10, eta=0.0001).fit(X, y)
# ax[1].plot(range(1, len(ada2.cost_)+1), ada2.cost_, marker='o')
# ax[1].set_xlabel('Epochs')
# ax[1].set_ylabel('Sum-squared-error')
# ax[1].set_title('Adaline - Learning rate 0.0001')

# plt.show()

X_std = np.copy(X)

X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

ada = AdalineGD(n_iter=15, eta=0.01)
ada.fit(X_std, y)

plot_decision_regions(X_std, y, classifier=ada)
plt.title('Adaline - Gradient Descent')
plt.xlabel('sepal length [standarized')
plt.ylabel('petal length [standarized')
plt.legend(loc='upper left')
plt.show()

plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Sum-squared-error')
plt.show()