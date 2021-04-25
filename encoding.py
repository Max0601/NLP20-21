import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
import numpy as np
from nltk.corpus import stopwords
from langdetect import detect
import string
from sklearn.feature_extraction.text import TfidfVectorizer

def split(s: str):
    return s.split(' ')

def remove_punctuation(s):
    return s.translate(str.maketrans('', '', string.punctuation))

def lower(s): #all into lowercase
    return s.lower()

def split_data(data):  #split the review into a list of words
    data['text'] = data['text'].map(lower)
    data['text'] = data['text'].map(remove_punctuation)
    data['text'] = data['text'].map(split)
    return data

def encoding_tfidf(data):
    v = TfidfVectorizer(stop_words='english', max_df=0.03)
    x = v.fit_transform(data['text'])
    return x
