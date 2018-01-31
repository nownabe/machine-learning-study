import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_wine = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data",
        header=None)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.fit_transform(X_test)

# Cov matrix
# cov_mat = np.cov(X_train_std.T)
# eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
# print('\nEigenavlues \n%s' % eigen_vals)

# tot = sum(eigen_vals)
# var_exp = [(i/tot) for i in sorted(eigen_vals, reverse=True)]
# cum_var_exp = np.cumsum(var_exp)

# plt.bar(range(1, 14), var_exp, alpha=0.5, align='center', label='individual explained variance')
# plt.step(range(1, 14), cum_var_exp, where='mid', label='cumulative explained variance')
# plt.ylabel('Explained variance ratio')
# plt.xlabel('Principal components')
# plt.legend(loc='best')
# plt.show()

# eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(eigen_vals))]
# eigen_pairs.sort(reverse=True)

# w = np.hstack((eigen_pairs[0][1][:, np.newaxis], eigen_pairs[1][1][:, np.newaxis]))
# print('Matrix W:\n', w)
# print(X_train_std[0].dot(w))
# X_train_pca = X_train_std.dot(w)

# colors = ['r', 'b', 'g']
# markers = ['s', 'x', 'o']
# for l, c, m in zip(np.unique(y_train), colors, markers):
#     plt.scatter(X_train_pca[y_train==1, 0], X_train_pca[y_train==1, 1], c=c, label=l, marker=m)
# plt.xlabel('PC 1')
# plt.ylabel('PC 2')
# plt.legend(loc='lower left')
# plt.show()

from my_funcs import plot_decision_regions
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

# pca = PCA(n_components=2)
# lr = LogisticRegression()
# X_train_pca = pca.fit_transform(X_train_std)
# X_test_pca = pca.transform(X_test_std)
# lr.fit(X_train_pca, y_train)
# plot_decision_regions(X_train_pca, y_train, classifier=lr)
# plt.xlabel('PC1')
# plt.ylabel('PC2')
# plt.legend(loc='lower left')
# plt.show()

# plot_decision_regions(X_test_pca, y_test, classifier=lr)
# plt.xlabel('PC1')
# plt.ylabel('PC2')
# plt.legend(loc='lower left')
# plt.show()

# np.set_printoptions(precision=4)
# mean_vecs = []
# for label in range(1, 4):
#     mean_vecs.append(np.mean(X_train_std[y_train==label], axis=0))
#     # print('MV %s: %s\n' %(label, mean_vecs[label-1]))

# d = 13 # # of features
# S_W = np.zeros((d, d))
# for label,mv in zip(range(1, 4), mean_vecs):
#     class_scatter = np.zeros((d, d))
#     for row in X_train_std[y_train == label]:
#         row, mv = row.reshape(d, 1), mv.reshape(d, 1)
#         class_scatter += (row-mv).dot((row-mv).T)
#     S_W += class_scatter
# print("Within-class scatter matrix: %sx%s" % (S_W.shape[0], S_W.shape[1]))

# d = 13 # # of features
# S_W = np.zeros((d, d))
# for label,mv in zip(range(1, 4), mean_vecs):
#     class_scatter = np.cov(X_train_std[y_train==label].T)
#     S_W += class_scatter
# print("Scaled within-class scatter matrix: %sx%s" % (S_W.shape[0], S_W.shape[1]))

# mean_overall = np.mean(X_train_std, axis=0)
# d = 13 # # of features
# S_B = np.zeros((d, d))
# for i, mean_vec in enumerate(mean_vecs):
#     n = X_train[y_train==i+1, :].shape[0]
#     mean_vec = mean_vec.reshape(d, 1)
#     mean_overall = mean_overall.reshape(d, 1)
#     S_B += n * (mean_vec - mean_overall).dot((mean_vec - mean_overall).T)
# print("Between-class scatter matrix: %sx%s" % (S_W.shape[0], S_W.shape[1]))

# eigen_vals, eigen_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
# eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(eigen_vals))]
# eigen_pairs = sorted(eigen_pairs, key=lambda k: k[0], reverse=True)
# print("Eigenvalues in decreasing order:\n")
# for eigen_val in eigen_pairs:
#    print(eigen_val[0])

# tot = sum(eigen_vals.real)
# discr = [(i / tot) for i in sorted(eigen_vals.real, reverse=True)]
# cum_discr = np.cumsum(discr)
# plt.bar(range(1, 14), discr, alpha=0.5, align='center', label='individual "discriminability"')
# plt.step(range(1, 14), cum_discr, where='mid', label='cumulative "discriminability"')
# plt.ylabel('"discriminability" ratio')
# plt.xlabel('Linear Discriminants')
# plt.ylim([-0.1, 1.1])
# plt.legend(loc='best')
# plt.show()

# w = np.hstack((eigen_pairs[0][1][:, np.newaxis].real, eigen_pairs[1][1][:, np.newaxis].real))
# print('Matrix W:\n', w)

# X_train_lda = X_train_std.dot(w)
# colors = ['r', 'b', 'g']
# markers = ['s', 'x', 'o']
# for l, c, m in zip(np.unique(y_train), colors, markers):
#     plt.scatter(X_train_lda[y_train==1, 0] * (-1), X_train_lda[y_train==1, 1] * (-1), c=c, label=l, marker=m)
# plt.xlabel('LD 1')
# plt.ylabel('LD 2')
# plt.legend(loc='lower right')
# plt.show()

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# lda = LinearDiscriminantAnalysis(n_components=2)
# X_train_lda = lda.fit_transform(X_train_std, y_train)
#
# lr = LogisticRegression()
# lr = lr.fit(X_train_lda, y_train)

# plot_decision_regions(X_train_lda, y_train, classifier=lr)
# plt.xlabel('LD 1')
# plt.ylabel('LD 2')
# plt.legend(loc='lower left')
# plt.show()

# X_test_lda = lda.transform(X_test_std)
# plot_decision_regions(X_test_lda, y_test, classifier=lr)
# plt.xlabel('LD 1')
# plt.ylabel('LD 2')
# plt.legend(loc='lower left')
# plt.show()

