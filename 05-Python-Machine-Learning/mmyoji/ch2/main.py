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

plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')

plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='uppper left')
plt.show()
