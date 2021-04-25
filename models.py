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

def logistic_model(data, encoding):
    x_train, x_test, y_train, y_test = train_test_split(data['text'].values, data['review_stars'].values,
                                                        test_size=0.25, random_state=1000)
    if encoding == 'CountVectorizer':
        vectorizer = CountVectorizer()
        vectorizer.fit(x_train)
        x_train = vectorizer.transform(x_train)
        x_test = vectorizer.transform(x_test)

    classifier = LogisticRegression(max_iter=1000000)
    classifier.fit(x_train, y_train)
    score = classifier.score(x_test, y_test)
    y_predicted = classifier.predict(x_test)
    print("Accuracy:", score)

def keras_model(data, encoding):
    x_train, x_test, y_train, y_test = train_test_split(data['text'].values, data['review_stars'].values,
                                                        test_size=0.25, random_state=1000)
    if encoding == 'vectorizer':
        vectorizer = CountVectorizer()
        vectorizer.fit(x_train)
        x_train = vectorizer.transform(x_train)
        x_test = vectorizer.transform(x_test)

    input_dim = x_train.shape[1]

    model = Sequential()
    model.add(layers.Dense(5, input_dim=input_dim, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.summary()

    history = model.fit(x_train, y_train,
                        epochs=50,
                        verbose=False,
                        validation_data=(x_test, y_test),
                        batch_size=100)
    loss, accuracy = model.evaluate(x_train, y_train, verbose=False)
    print("Training Accuracy: {:.4f}".format(accuracy))
    loss, accuracy = model.evaluate(x_test, y_test, verbose=False)
    print("Testing Accuracy:  {:.4f}".format(accuracy))

