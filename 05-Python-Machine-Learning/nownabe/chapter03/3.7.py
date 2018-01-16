import sys, os
sys.path.append(os.pardir)
from util import plot_decision_regions

import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
X_combined_std = np.vstack((X_train_std, X_test_std))

def s3_7():
    knn = KNeighborsClassifier(n_neighbors=5, p=2, metric="minkowski")
    knn.fit(X_train_std, y_train)
    plot_decision_regions(X_combined_std, y_combined, classifier=knn, test_idx=range(105, 150))
    plt.xlabel("petal length [standardized]")
    plt.ylabel("petal width [standardized]")
    plt.legend(loc="upper left")
    plt.show()


s3_7()
