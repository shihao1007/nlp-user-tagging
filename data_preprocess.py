#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 23:28:06 2019

Preprocess for raw Twitter data
The data can be found in ./dataframes folder
The preprocess including a tokenization function des_tokenize()
followed by removing null values and label the gamer column
using gamers as class 1, and programmers as class 0

Author:
    Shihao Ran
    shihao1007@gmail.com
    STIM Laboratory
"""
# import packages
# pandas for manupulating dataframes
import pandas as pd
# regular expression package for pattern-matching
import re
# nltk, natural language toolkit
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# initialize a lemmatizer
# A lemmatizer switch all the words in different forms but same meanings
# back to their lemma form (root form)
# e.g. dogs, doggy -> dog
word_lemmatizer = WordNetLemmatizer()

# define a udf to apply to the whole dataframe
def preprocess(row):
    """
    preprocessing for the description column:
        1. lowercasing all the words
        2. tokenize the sentense
        3. keep only the alphabetical words
        4. remove all stopwords
        5. lemmatize all tokens

    Parameters
    ----------
        row: row of a pandas dataframe
            current row of the dataframe

    Returns
    -------
        processed: string
            Note that the return is not a list of tokens,
            intead, the list is converted back to a string
            for later use for sklearn CountVectorizer
    """
    # lowercasing, tokenization, and keep only alphabetical tokens
    tokens = [word for word in word_tokenize(row.lower())\
              if word.isalpha()]
    # filtering out tokens that are not all alphabetical
    tokens = [word for word in re.findall(r'[A-Za-z]+', ' '.join(tokens))]
    # remove all stopwords
    no_stop = [word for word in tokens\
               if word not in stopwords.words('english')]
    # lematizing all tokens
    lemmatized = [word_lemmatizer.lemmatize(word) for word in no_stop]
    # convert tokens back to a sentense as the input for CountVectorizer later
    processed = ' '.join(lemmatized)

    # return the clean sentense
    return processed

def __main__():
    
    # load all raw data
    df = pd.read_csv(r'./dataframes/user_and_description.csv')
    # subset for experiment
    #df = temp.iloc[:100, :].copy()
    # clean the description column
    df['description'] = df.description.apply(preprocess)
    # remove null rows
    df = df[df.description != '']
    # create training label column
    df['gamer'] = df.user_type.apply(lambda x: 1 if x=='gamer' else 0)
    
    # save the processed dataframe
    df.to_csv(r'./dataframes/processed_data.csv')
