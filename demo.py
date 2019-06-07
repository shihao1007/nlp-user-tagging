# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 22:16:26 2019

Automated Twitter User Tagging using NLP
This is a demo using pre-trained Linear SVC model

Author:
    Shihao Ran
    shihao1007@gmail.com
    STIM Laboratory
"""

from joblib import load
from data_preprocess import preprocess

# load Linear SVC model
lc = load(r'./models/LinearSVC.pkl')

# use this function to predict a string of description
def user_tagging(des):
    des_in = preprocess(des)
    pred = lc.predict([des_in])
    if pred == [1]:
        print("This user is a gamer")
    elif pred == [0]:
        print("This user is a programmer")
    else:
        raise ValueError("Invalid prediction.")
