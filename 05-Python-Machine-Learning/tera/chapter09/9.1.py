import os
import pickle
import re
from movieclassifier.vectorizer import vect
import numpy as np

os.chdir('movieclassifier')

clf = pickle.load(open(os.path.join('pkl_objects', 'classifier.pkl'), 'rb'))

label = {0:'negative', 1:'positive'}

example = ['I love this movie']
X = vect.transform(example)
print('Prediction: %s\nProbability: %.2f%%' %\
      (label[clf.predict(X)[0]], clf.predict_proba(X).max()*100))
