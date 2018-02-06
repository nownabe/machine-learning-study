import pyprind
import pandas as pd
import os
import numpy as np
import re

from nltk.stem.porter import PorterStemmer

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

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

def tokenizer(text):
    return text.split()

porter = PorterStemmer()

def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]

stop = stopwords.words('english')

X_train = df.loc[:25000, 'review'].values
y_train = df.loc[:25000, 'sentiment'].values
X_test = df.loc[25000:, 'review'].values
y_test = df.loc[25000:, 'sentiment'].values

tfidf = TfidfVectorizer(strip_accents=None, lowercase=False, preprocessor=None)
param_grid = [{'vect__ngram_range': [(1, 1)],
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [tokenizer, tokenizer_porter],
               'clf__penalty': ['l1', 'l2'],
               'clf__C': [1.0, 10.0, 100.0]},
              {'vect__ngram_range': [(1, 1)],
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [tokenizer, tokenizer_porter],
               'vect__use_idf': [False],
               'vect__norm': [None],
               'clf__penalty': ['l1', 'l2'],
               'clf__C': [1.0, 10.0, 100.0]}]
lr_tfidf = Pipeline([('vect', tfidf),
                     ('clf', LogisticRegression(random_state=0))])
gs_lr_tfidf = GridSearchCV(lr_tfidf, param_grid, scoring='accuracy', cv=5, verbose=1, n_jobs=-1)
gs_lr_tfidf.fit(X_train, y_train)

print('Best parameter set: %s ' % gs_lr_tfidf.best_params_)
print('CV Accuracy: %.3f' % gs_lr_tfidf.best_score_)

clf = gs_lr_tfidf.best_estimator_
print('Test Accuracy: %.3f' % clf.score(X_test, y_test))

