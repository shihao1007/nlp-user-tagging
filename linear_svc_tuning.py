# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 11:44:27 2019

Grid Search and Cross-Validation for LinearSVC model

Author:
    Shihao Ran
    shihao1007@gmail.com
    STIM Laboratory
"""

# import packages
import pandas as pd
from pprint import pprint
from time import time
import logging
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
from joblib import dump

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn import metrics

# performance visualization tools
from yellowbrick.classifier import ClassPredictionError
from yellowbrick.classifier import ClassificationReport
from yellowbrick.classifier import ROCAUC
from yellowbrick.classifier import ConfusionMatrix

print(__doc__)

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
# load dataset
df = pd.read_csv(r'./dataframes/processed_data.csv')
X = df['description'].values.astype('U')
y = df['gamer'].values.astype('U')
X, y = RandomOverSampler(random_state=5).fit_sample(X.reshape(-1, 1), y)
X = pd.Series(X.reshape(X.shape[0]))
# split into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=.2,
                                                    random_state=5)
X_train = X_train.values.astype('U')
X_test = X_test.values.astype('U')

# initialize the pipeline
pipeline = Pipeline([
    ('vect', CountVectorizer(ngram_range=(1, 2))),
    ('tfidf', TfidfTransformer(use_idf=True)),
    ('lc', LinearSVC(dual=False)),
])

# specify parameter grid to search over
parameters = {
    'vect__max_df': (0.25, 0.5),
    'lc__penalty': ('l1', 'l2'),
    'lc__C': (0.5, 0.75, 1)
}

# initialize the GridSearchCV object
grid_search = GridSearchCV(pipeline, parameters, cv=5,
                           n_jobs=8, verbose=10)

# print progress to the console
print("Performing grid search...")
print("pipeline:", [name for name, _ in pipeline.steps])
print("parameters:")
pprint(parameters)
t0 = time()
# start searching
grid_search.fit(X_train, y_train)
print("done in %0.3fs" % (time() - t0))
print()
# print the best paramters found
print("Best score: %0.3f" % grid_search.best_score_)
print("Best parameters set:")
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))

# evaluate the best model
for viz in [ClassificationReport, ClassPredictionError,
                ROCAUC, ConfusionMatrix]:
        if viz == ROCAUC:
            # print current model and visualization combination
            print(grid_search.best_estimator_, viz)
            # fit and evaluate the model
            viz1 = viz(grid_search.best_estimator_, micro=False, macro=False, per_class=False)
            viz1.fit(X_train, y_train.astype(int))
            viz1.score(X_test, y_test.astype(int))
            plt.figure()
            viz1.poof()
        else:
            # print current model and visualization combination
            print(grid_search.best_estimator_, viz)
            # fit and evaluate the model
            viz1 = viz(pipeline)
            viz1.fit(X_train, y_train.astype(int))
            viz1.score(X_test, y_test.astype(int))
            plt.figure()
            viz1.poof()

# save the best model
best_model = grid_search.best_estimator_
dump(best_model, './models/LinearSVC.pkl')
