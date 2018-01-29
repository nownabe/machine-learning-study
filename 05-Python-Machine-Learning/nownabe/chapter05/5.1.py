import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from util import plot_decision_regions

df_wine = pd.read_csv("../data/wine.data", header=None)

X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)


def s5_1_1():
    cov_mat = np.cov(X_train_std.T)
    eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
    print("\nEigenvalues \n%s" % eigen_vals)
    tot = sum(eigen_vals)
    var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
    cum_var_exp = np.cumsum(var_exp)
    plt.bar(range(1, 14), var_exp, alpha=0.5, align="center", label="individual explained variance")
    plt.step(range(1, 14), cum_var_exp, where="mid", label="cumulative explained variance")
    plt.ylabel("Explained variance ratio")
    plt.xlabel("Principal components")
    plt.legend(loc="best")
    plt.show()


def s5_1_2():
    cov_mat = np.cov(X_train_std.T)
    eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
    eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(eigen_vals))]
    eigen_pairs.sort(key=lambda k: k[0], reverse=True)
    w = np.hstack((eigen_pairs[0][1][:, np.newaxis], eigen_pairs[1][1][:, np.newaxis]))
    print("Matrix W:\n", w)
    print(X_train_std[0].dot(w))
    X_train_pca = X_train_std.dot(w)

    colors = ["r", "b", "g"]
    markers = ["s", "x", "o"]
    for l, c, m in zip(np.unique(y_train), colors, markers):
        plt.scatter(X_train_pca[y_train == l, 0], X_train_pca[y_train == l, 1], c=c, label=l, marker=m)

    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.legend(loc="lower left")
    plt.show()


def s5_1_3():
    pca = PCA(n_components=2)
    lr = LogisticRegression()
    X_train_pca = pca.fit_transform(X_train_std)
    X_test_pca = pca.transform(X_test_std)
    lr.fit(X_train_pca, y_train)
    plot_decision_regions(X_train_pca, y_train, classifier=lr)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(loc="lower left")
    plt.show()

    plot_decision_regions(X_test_pca, y_test, classifier=lr)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(loc="lower left")
    plt.show()

    pca = PCA(n_components=None)
    X_train_pca = pca.fit_transform(X_train_std)
    print(pca.explained_variance_ratio_)



s5_1_1()
s5_1_2()
s5_1_3()
