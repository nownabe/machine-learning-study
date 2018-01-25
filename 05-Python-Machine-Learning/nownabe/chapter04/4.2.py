import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

df = pd.DataFrame([
    ["green", "M", 10.1, "class1"],
    ["red", "L", 13.5, "class2"],
    ["blue", "XL", 15.3, "class1"],
])

df.columns = ["color", "size", "price", "classlabel"]


def s4_2():
    print("==== 4.2 ====")
    print(df)


def s4_2_1():
    print("==== 4.2.1 ====")

    size_mapping = {"XL": 3, "L": 2, "M": 1}
    df["size"] = df["size"].map(size_mapping)
    print(df)


def s4_2_2():
    print("==== 4.2.2 ====")

    class_mapping = {label: idx for idx, label in enumerate(np.unique(df["classlabel"]))}
    print(class_mapping)

    df["classlabel"] = df["classlabel"].map(class_mapping)
    print(df)

    inv_class_mapping = {v: k for k, v in class_mapping.items()}
    df["classlabel"] = df["classlabel"].map(inv_class_mapping)
    print(df)

    class_le = LabelEncoder()
    y = class_le.fit_transform(df["classlabel"].values)
    print(y)

    print(class_le.inverse_transform(y))


def s4_2_3():
    print("==== 4.2.3 ====")

    X = df[["color", "size", "price"]].values
    color_le = LabelEncoder()
    X[:, 0] = color_le.fit_transform(X[:, 0])
    print(X)

    ohe = OneHotEncoder(categorical_features=[0])
    print(ohe.fit_transform(X).toarray())

    print(pd.get_dummies(df[["price", "color", "size"]]))


s4_2()
s4_2_1()
s4_2_2()
s4_2_3()
