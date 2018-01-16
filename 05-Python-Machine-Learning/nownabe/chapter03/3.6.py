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

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))

def gini(p):
    return p * (1 - p) + (1 - p) * (1 - (1 - p))

def entropy(p):
    return - p * np.log2(p) - (1 - p) * np.log2(1 - p)

def error(p):
    return 1 - np.max([p, 1 - p])

def s3_6_1():
    x = np.arange(0.0, 1.0, 0.01)
    ent = [entropy(p) if p != 0 else None for p in x]
    sc_ent = [e * 0.5 if e else None for e in ent]
    err = [error(i) for i in x]
    fig = plt.figure()
    ax = plt.subplot(111)

    labels = ["Entropy", "Entropy (scaled)", "Gini Impurity", "Misclassification Error"]
    lines  = ["-", "-", "--", "-."]
    colors = ["black", "lightgray", "red", "green", "cyan"]

    for i, lab, ls, c, in zip([ent, sc_ent, gini(x), err], labels, lines, colors):
        line = ax.plot(x, i, label=lab, linestyle=ls, lw=2, color=c)

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=False)
    ax.axhline(y=0.5, linewidth=1, color="k", linestyle="--")
    ax.axhline(y=1.0, linewidth=1, color="k", linestyle="--")
    plt.ylim([0, 1.1])
    plt.xlabel("p(i=1)")
    plt.ylabel("Impurity Index")
    plt.show()


def s3_6_2():
    tree = DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=0)
    tree.fit(X_train, y_train)
    plot_decision_regions(X_combined, y_combined, classifier=tree, test_idx=range(105, 150))
    plt.xlabel("petal length [cm]")
    plt.ylabel("petal width [cm]")
    plt.legend(loc="upper left")
    plt.show()
    export_graphviz(tree, out_file="tree.dot", feature_names=["petal length", "petal width"])


def s3_6_3():
    rf = RandomForestClassifier(criterion="entropy", n_estimators=10, random_state=1, n_jobs=2)
    rf.fit(X_train, y_train)
    plot_decision_regions(X_combined, y_combined, classifier=rf, test_idx=range(105, 150))
    plt.xlabel("petal length [cm]")
    plt.ylabel("petal width [cm]")
    plt.legend(loc="upper left")
    plt.show()


s3_6_1()
s3_6_2()
s3_6_3()
