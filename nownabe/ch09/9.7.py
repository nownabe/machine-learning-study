import numpy as np
from math import factorial

def poisson(l):
    def f(x):
        return np.exp(-l) * (l ** x) / factorial(x)

    return f

def cumulative(dist, r):
    sum = 0
    for x in r:
        sum += dist(x)

    return sum

table = {
    "北海道": {
        "deaths": 9.7,
        "casualties": 526.6,
    },
    "東京": {
        "deaths": 4.0,
        "casualties": 508.7,
    },
    "大阪": {
        "deaths": 5.7,
        "casualties": 703.8,
    },
    "福岡": {
        "deaths": 7.8,
        "casualties": 867.2,
    },
}

print("(1) 1年間の交通事故死亡者数が10人未満である確率")

for pref in table.keys():
    f = poisson(table[pref]["deaths"])
    print("{0}: {1:0.3f}".format(pref, cumulative(f, range(10))))

print()

print("(2) 1日の交通事故死傷者数が5人未満である確率")

for pref in table.keys():
    # 1年のうちでそれぞれの日の交通事故死傷者数が同じ確率分布に従うとする
    f = poisson(table[pref]["casualties"] / 365)
    print("{0}: {1:0.3f}".format(pref, cumulative(f, range(5))))
