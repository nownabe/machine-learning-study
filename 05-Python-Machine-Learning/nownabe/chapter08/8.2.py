import nltk
import numpy as np
import pandas as pd
import re

from nltk import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

np.set_printoptions(precision=2)

docs = np.array([
    "The sun is shining",
    "The weather is sweet",
    "The sun is shining, the weather is sweet, and one and one is two"
])

df = pd.read_csv("../data/movie_data.csv")

porter = PorterStemmer()

nltk.download("stopwords")


def s8_2_1():
    count = CountVectorizer()
    bag = count.fit_transform(docs)
    print(count.vocabulary_)
    print(bag.toarray())


def s8_2_2():
    count = CountVectorizer()
    tfidf = TfidfTransformer(use_idf=True, norm="l2", smooth_idf=True)
    print(tfidf.fit_transform(count.fit_transform(docs)).toarray())


def preprocessor(text):
    text = re.sub("<[^>]*>", "", text)
    emoticons = re.findall("(?::|;|=)(?:-)?(?:\)|\(|D|P)", text)
    text = re.sub("[\W]+", " ", text.lower()) + " ".join(emoticons).replace("-", "")
    return text


def s8_2_3():
    print(df.loc[0, "review"][-50:])
    print(preprocessor(df.loc[0, "review"][-50:]))
    print(preprocessor("</a>This :) is :( a test :-)!"))
    df["review"] = df["review"].apply(preprocessor)


def tokenizer(text):
    return text.split()


def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]


def s8_2_4():
    print(tokenizer("runners like running and thus they run"))
    print(tokenizer_porter("runners like running and thus they run"))

    stop = stopwords.words("english")
    ws = [w for w in tokenizer_porter("a runner likes running and runs a lot")[-10:] if w not in stop]
    print(ws)


def s8_2_5():
    X_train = df.loc[:25000, "review"].values
    y_train = df.loc[:25000, "sentiment"].values
    X_test = df.loc[25000:, "review"].values
    y_test = df.loc[25000:, "sentiment"].values

    tfidf = TfidfVectorizer(strip_accents=None, lowercase=False, preprocessor=None)
    stop = stopwords.words("english")
    param_grid = [{
        "vect__ngram_range": [(1, 1)],
        "vect__stop_words": [stop, None],
        "vect__tokenizer": [tokenizer, tokenizer_porter],
        "clf__penalty": ["l1", "l2"],
        "clf__C": [1.0, 10.0, 100.0]
    }, {
        "vect__ngram_range": [(1, 1)],
        "vect__stop_words": [stop, None],
        "vect__tokenizer": [tokenizer, tokenizer_porter],
        "vect__use_idf": [False],
        "vect__norm": [None],
        "clf__penalty": ["l1", "l2"],
        "clf__C": [1.0, 10.0, 100.0]
    }]
    lr_tfidf = Pipeline([("vect", tfidf), ("clf", LogisticRegression(random_state=0))])
    gs = GridSearchCV(lr_tfidf, param_grid, scoring="accuracy", cv=5, verbose=1, n_jobs=-1)
    gs.fit(X_train, y_train)
    print("Best parameter set: %s" % gs.best_params_)
    print("CV Accuracy: %.3f" % gs.best_score_)
    clf = gs.best_estimator_
    print("Test Accuracy: %.3f" % clf.score(X_test, y_test))


s8_2_1()
s8_2_2()
s8_2_3()
s8_2_4()
s8_2_5()
