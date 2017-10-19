import itertools
import numpy as np


def sample_variance(data):
    return ((data - data.mean()) ** 2).sum() / (data.size - 1)


data = np.array([171.0, 167.3, 170.6, 178.7, 162.3])
n = 3
sample_means = []

print("母集団: ", end="")
print(data)

print("\n(1) この母集団の母平均を求めよ")
print(data.mean())

print(f"\n(2) {n}人を標本とするとき、取り得る標本をすべて書き出し、標本平均と標本分散を計算せよ")
for samples in itertools.combinations(data, n):
    samples = np.array(list(samples))
    mean = samples.mean()
    sample_means.append(mean)
    print("Samples: ", samples, end="")
    print(", mean={0:0.2f}".format(mean), end="")
    print(", variance={0:0.2f}".format(sample_variance(samples)))

print("\n(3) 標本平均の確率分布、期待値、分散を(2)の結果から求めよ")
sample_means = np.array(sample_means)
variance = ((sample_means - sample_means.mean()) ** 2).sum() / sample_means.size
print("期待値: ", sample_means.mean())
print("分散  : ", variance)
print()

variance = ((data - data.mean()) ** 2).sum() / data.size
print("母分散: ", variance)
print("(9.11): ", (data.size - n) / (data.size - 1) * variance / n)
