from io import StringIO

import pandas as pd

from sklearn.preprocessing import Imputer



csv_data = '''A,B,C,D
1.0,2.0,3.0,4.0
5.0,6.0,,8.0
10.0,11.0,12.0,'''

df = pd.read_csv(StringIO(csv_data))


def s4_1():
    print(df)
    print("---")
    print(df.isnull())
    print("---")
    print(df.isnull().sum())


def s4_1_1():
    print(df.dropna())
    print("---")
    print(df.dropna(axis=1))


def s4_1_2():
    imr = Imputer(missing_values="NaN", strategy="mean", axis=0)
    imr = imr.fit(df)
    imputed_data = imr.transform(df.values)
    print(imputed_data)


s4_1()
s4_1_1()
s4_1_2()
