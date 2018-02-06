import pyprind
import pandas as pd
import os
import numpy as np

CSV_FILE = './movie_data.csv'

# # Convert aclImdb files into a CSV file
# pbar = pyprind.ProgBar(50000)
# labels = {'pos': 1, 'neg': 0}
# df = pd.DataFrame()
#
# for s in ('test', 'train'):
#     for l in ('pos', 'neg'):
#         path = './aclImdb/%s/%s' % (s, l)
#
#         for file in os.listdir(path):
#             with open(os.path.join(path, file), 'r', encoding='utf-8') as infile:
#                 txt = infile.read()
#
#             df = df.append([[txt, labels[l]]], ignore_index=True)
#             pbar.update()
#
# df.columns = ['review', 'sentiment']
#
# np.random.seed(0)
#
# df = df.reindex(np.random.permutation(df.index))
# df.to_csv(CSV_FILE, index=False)

df = pd.read_csv(CSV_FILE)
# # Check CSV is whether correct or not
# print(df.head(3))
