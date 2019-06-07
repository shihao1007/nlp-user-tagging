# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 11:44:27 2019

loop through the predefined model pool to find
the best model (with all default hyperparameters)

Author:
    Shihao Ran
    shihao1007@gmail.com
    STIM Laboratory
"""
# import packages
import logging
import pandas as pd
import matplotlib.pyplot as plt
from pprint import pprint
from time import time

# Over sampling and under sampling tool for imbalanced data sets
from imblearn.over_sampling import RandomOverSampler

# scikit-learn
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

# classification models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

# performance visualization tools
from yellowbrick.classifier import ClassPredictionError
from yellowbrick.classifier import ClassificationReport
from yellowbrick.classifier import ROCAUC
from yellowbrick.classifier import ConfusionMatrix

print(__doc__)
# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

# laod in dataset
df = pd.read_csv(r'./dataframes/processed_data.csv')

# seperate features and labels
X = df['description'].values.astype('U')
y = df['gamer'].values.astype('U')

# balance the number of samples for each class
X, y = RandomOverSampler().fit_sample(X.reshape(-1, 1), y)
X = pd.Series(X.reshape(X.shape[0]))

# split into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=.2,
                                                    random_state=5)

# change data type of features to fit in CountVectorizer
X_train = X_train.values.astype('U')
X_test = X_test.values.astype('U')

# initialize benchmark paramters
scores = []
training_time = []
test_time = []
# train and visualize the results of each model individually
for model in [LogisticRegression, RandomForestClassifier, LinearSVC,
              KNeighborsClassifier, MultinomialNB]:
    # instantiate a pipeline
    pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('model', model()),
    ])
    # print the current model in the pipeline
    print(model)
    # for four different metrics and visualization
    for viz in [ClassificationReport, ClassPredictionError,
                ROCAUC, ConfusionMatrix]:
        # use the ClassificationReport to do profiling
        if viz == ClassificationReport:
            print(model)
            # instantiate yellowbrick visualization object
            viz1 = ClassificationReport(pipeline)
            # start a timer before the training
            t1 = time()
            # train the model
            viz1.fit(X_train, y_train.astype(int))
            # save training time
            training_time.append(time()-t1)
            # reset the timer
            t1 = time()
            # evaluate the model
            score = viz1.score(X_test, y_test.astype(int))
            # savde test time
            test_time.append(time()-t1)
            # save overall model score
            scores.append(score)
            # show the visualization
            plt.figure()
            viz1.poof()

        # a special case for LinearSVC and ROC curve
        elif viz == ROCAUC and model == LinearSVC:
            # print current model and visualization combination
            print(model, viz)
            # only plot the curve for the samples
            viz1 = viz(pipeline, micro=False, macro=False, per_class=False)
            viz1.fit(X_train, y_train.astype(int))
            viz1.score(X_test, y_test.astype(int))
            plt.figure()
            viz1.poof()

        # for anyother combinations
        else:
            print(model, viz)
            viz1 = viz(pipeline)
            viz1.fit(X_train, y_train.astype(int))
            viz1.score(X_test, y_test.astype(int))
            plt.figure()
            viz1.poof()
