import pickle
import re
import os
import numpy as np

from vectorizer import vect

CLASSIFIER = 'classfier.pkl' # typo

clf = pickle.load(open(os.path.join('pkl_objects', CLASSIFIER), 'rb'))

label = {0: 'negative', 1: 'positive'}
example = ['I love this movie']
X = vect.transform(example)
print('Prediction: %s\nProbability: %.2f%%' % (label[clf.predict(X)[0]], np.max(clf.predict_proba(X)) * 100))
