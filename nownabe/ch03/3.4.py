import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

count = 200
# count = 20000

bro = [71, 68, 66, 67, 70, 71, 70, 73, 72, 65, 66]
sis = [69, 64, 65, 63, 65, 62, 65, 64, 66, 59, 62]

df = pd.DataFrame([bro, sis], index=["Brother", "Sister"])

rs = []

for c in range(20000):
    indexes = []
    for _ in range(11):
        indexes.append(random.randrange(11))

    pieces = []
    for i in indexes:
        pieces.append(df[i])

    data = pd.concat(pieces, axis=1, ignore_index=True)
    r = np.corrcoef(data.T["Brother"], data.T["Sister"])[0, 1]
    rs.append(r)

# スタージェスの公式 p. 22
k = 1 + round(math.log2(len(rs)))

plt.hist(rs, bins=k)
plt.show()
