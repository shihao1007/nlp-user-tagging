# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 11:44:27 2019

Grid Search and Cross-Validation for Naive Bayes model
Please refer to linear_svc_tuning.py for detailed comments

Author:
    Shihao Ran
    shihao1007@gmail.com
    STIM Laboratory
"""

import pandas as pd
from pprint import pprint
from time import time
import logging
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn import metrics

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB

from yellowbrick.classifier import ClassPredictionError
from yellowbrick.classifier import ClassificationReport
from yellowbrick.classifier import ROCAUC
from yellowbrick.classifier import ConfusionMatrix

from imblearn.over_sampling import RandomOverSampler

print(__doc__)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

df = pd.read_csv(r'./dataframes/processed_data.csv')
X = df['description'].values.astype('U')
y = df['gamer'].values.astype('U')
X, y = RandomOverSampler(random_state=5).fit_sample(X.reshape(-1, 1), y)
X = pd.Series(X.reshape(X.shape[0]))
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=.2,
                                                    random_state=5)
X_train = X_train.values.astype('U')
X_test = X_test.values.astype('U')
#%%
pipeline = Pipeline([
    ('vect', CountVectorizer(ngram_range=(1, 2))),
    ('tfidf', TfidfTransformer(use_idf=True)),
    ('nb', MultinomialNB(fit_prior=False)),
])
parameters = {
    'vect__max_df': (0.25, 0.5),
    'nb__alpha': (0, 0.1, 0.2, 0.3, 0.4)
}
grid_search = GridSearchCV(pipeline, parameters, cv=5,
                           n_jobs=8, verbose=10)
print("Performing grid search...")
print("pipeline:", [name for name, _ in pipeline.steps])
print("parameters:")
pprint(parameters)
t0 = time()
grid_search.fit(X_train, y_train)
print("done in %0.3fs" % (time() - t0))
print()
print("Best score: %0.3f" % grid_search.best_score_)
print("Best parameters set:")
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))

for viz in [ClassificationReport, ClassPredictionError,
            ROCAUC, ConfusionMatrix]:
    print(grid_search.best_estimator_, viz)
    viz1 = viz(grid_search.best_estimator_)
    viz1.fit(X_train, y_train)
    viz1.score(X_test, y_test)
    plt.figure()
    viz1.poof()
