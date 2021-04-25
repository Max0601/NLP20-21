import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
import numpy as np
from nltk.corpus import stopwords
from langdetect import detect
from encoding import split_data, encoding_tfidf
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from keras.models import Sequential
from keras import layers
from keras.backend import clear_session
from models import logistic_model, keras_model

## import dataset


data = pd.read_csv('data/sampledata.csv')

## Logistic regression with CountVectorizer

logistic_model(data, "CountVectorizer")

## Keras with countVectorizer

keras_model(data, "CountVectorizer")