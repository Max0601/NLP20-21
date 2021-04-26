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
import pandas as pd
import numpy as np
import seaborn as sns
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import matplotlib.pyplot as plt

def logistic_model(data, encoding):
    print('logistic model')
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
    clear_session()
    print("Keras")
    x_train, x_test, y_train, y_test = train_test_split(data['text'].values, data['review_stars'].values,
                                                        test_size=0.25, random_state=1000)
    if encoding == 'CountVectorizer':
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

def LSTM_model(data):
    clear_session()
    print("LSTM model")
    max_feature = 1000  #  number rows in embedding vector
    max_length = 100  # max words in a review

    tokenizer = Tokenizer(num_words=max_feature)
    tokenizer.fit_on_texts(data['text'])
    sequences = tokenizer.texts_to_sequences(data['text'])

    word_index = tokenizer.word_index

    x = pad_sequences(sequences, maxlen=max_length)
    y = to_categorical(np.asarray(data['review_stars'])-1)

    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        test_size=0.25, random_state=1000)

    # Glove Embedding

    GLOVE_DIR = 'data'

    import os
    embeddings_index = {}
    f = open(os.path.join(GLOVE_DIR, 'glove.6B.50d.txt'),encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    # Embedding matrix

    EMBEDDING_DIM = 50  # how big is each word vector

    embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector


    # Embedding layer

    from keras.layers import Embedding

    embedding_layer = Embedding(len(word_index) + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=max_length,
                                trainable=False)

    # training

    from keras.layers import Bidirectional, GlobalMaxPool1D, Conv1D
    from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation

    from keras.models import Model

    inp = Input(shape=(max_length,))
    x = embedded_sequences = embedding_layer(inp)
    x = Bidirectional(LSTM(50, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(x)
    x = GlobalMaxPool1D()(x)
    x = Dense(50, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(5, activation="softmax")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    history = model.fit(x_train, y_train,
                        epochs=20,
                        verbose=2,
                        validation_data=(x_test, y_test),
                        batch_size=100)

