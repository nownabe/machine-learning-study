import pyprind
import pandas as pd
import os
import numpy as np
import re

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

# Remove HTML markup in `text`
def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower()) + ''.join(emoticons).replace('-', '')
    return text

# # Test the function
# print(preprocessor(df.loc[0, 'review'][-50:]))
# print(preprocessor("</a>This :) is <br/>:( a test :-)!<br><br>"))

df['review'] = df['review'].apply(preprocessor)
