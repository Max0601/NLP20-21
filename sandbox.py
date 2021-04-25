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
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.callbacks import EarlyStopping

## import dataset
data = pd.read_csv('data/sampledata.csv')


# Encoding methods to transform the feature "reviews" str into for ex: One hot, etc as described in the initial report
#data = split_data(data)
x_train, x_test, y_train, y_test = train_test_split(data['text'].values, data['review_stars'].values, test_size=0.25, random_state=1000)

vectorizer = CountVectorizer()
vectorizer.fit(x_train)
x_train = vectorizer.transform(x_train)
x_test = vectorizer.transform(x_test)

#encoder = LabelEncoder()
#encoder.fit(y_train)
#encoded_Y = encoder.transform(y_train)


# Logistic regression

classifier = LogisticRegression(max_iter= 1000000)
classifier.fit(x_train, y_train)
score = classifier.score(x_test, y_test)
y_predicted = classifier.predict(x_test)
print("Accuracy:", score)


# Keras

y_train = y_train - 1
y_test = y_test - 1
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)


input_dim = x_train.shape[1]

model = Sequential()
model.add(layers.Dense(15, input_dim=input_dim, activation='relu'))
model.add(layers.Dropout(0.4))
model.add(layers.Dense(5, activation='softmax'))
model.compile(loss='categorical_crossentropy',
               optimizer='adam',
               metrics=['accuracy'])
model.summary()

es_callback = EarlyStopping(monitor='val_loss', patience=10)
history = model.fit(x_train, y_train,
                     epochs=50,
                     verbose=2,
                     validation_data=(x_test, y_test),
                     batch_size=100,
                     callbacks=[es_callback])


loss, accuracy = model.evaluate(x_train, y_train, verbose=2)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(x_test, y_test, verbose=2)
print("Testing Accuracy:  {:.4f}".format(accuracy))


clear_session()
# Learning methods to use the encoding feature and to classify the review, predict the number of stars based on the encoding feature
