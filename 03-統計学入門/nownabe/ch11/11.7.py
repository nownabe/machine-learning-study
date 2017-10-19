import numpy as np


def sample_variance(x):
    return ((x - x.mean()) ** 2).sum() / (x.size - 1)


x = np.array([21.8, 22.4, 22.7, 24.5, 25.9, 24.9, 24.8, 25.3, 25.2, 24.6])

print("\n(1) 母平均の信頼係数99%の信頼区間を求めよ")

x_s2 = sample_variance(x)
x_s  = np.sqrt(x_s2)
t    = 3.250 # 付表2より

lower = x.mean() - t * x_s / np.sqrt(x.size)
upper = x.mean() + t * x_s / np.sqrt(x.size)

print(f"東京: {x}")
print(f"a: 0.01")

print("東京の標本平均: {0:0.4f}".format(x.mean()))
print("東京の標本分散: {0:0.4f}".format(x_s2))
print("東京の標本偏差: {0:0.4f}".format(x_s))
print("t(9)の上側確率0.5%点: {0:0.3f}".format(t))

print("\n式(11.43)より")
print("信頼区間: [{0:0.4f}, {1:0.4f}]".format(lower, upper))

print("\n(2) 母分散の信頼係数95%の信頼区間を求めよ")

# 付表3より
kh_l = 19.0228
kh_u = 2.70039

lower = (x.size - 1) * x_s2 / kh_l
upper = (x.size - 1) * x_s2 / kh_u

print(f"χ(9)の上側確率2.5%点: {kh_l}")
print(f"χ(9)の下側確率2.5%点: {kh_u}")

print("\n式(11.45)より")
print("信頼区間: [{0:0.4f}, {1:0.4f}]".format(lower, upper))

print("\n(3) 東京と大阪の最低気温差の95%信頼区間を作れ")

y = np.array([22.1, 25.3, 23.3, 25.2, 25.3, 24.9, 24.9, 24.9, 24.9, 24.0])

d = x - y

print(f"大阪: {y}")
print(f"東京 - 大阪: {d}")

d_s2 = sample_variance(d)
d_s = np.sqrt(d_s2)
t = 2.262 # 付表2より

lower = d.mean() - t * d_s / np.sqrt(d.size)
upper = d.mean() + t * d_s / np.sqrt(d.size)

print("(東京 - 大阪)の標本平均: {0:0.4f}".format(d.mean()))
print("(東京 - 大阪)の標本分散: {0:0.4f}".format(d_s2))
print("(東京 - 大阪)の標本偏差: {0:0.4f}".format(d_s))
print("t(9)の上側確率2.5%点: {0:0.3f}".format(t))

print("\n式(11.43)より")
print("信頼区間: [{0:0.4f}, {1:0.4f}]".format(lower, upper))
