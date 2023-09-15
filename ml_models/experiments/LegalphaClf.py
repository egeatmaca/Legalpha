import numpy as np
import os
import pickle
from sklearn.base import BaseEstimator, ClassifierMixin
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Embedding, Bidirectional, LSTM
from sklearn.preprocessing import OneHotEncoder

class LegalphaClf:
    model_folder = 'tf_model'
    one_hot_encoder_file = 'tf_one_hot_encoder.pkl'

    def __init__(self, max_words=10000, max_len=500):
        self.max_words = max_words
        self.max_len = max_len
        self.tokenizer = Tokenizer(num_words=self.max_words)
        self.one_hot_encoder = OneHotEncoder()

    def fit(self, X, y, epochs=25, batch_size=128, validation_split=None):
        self.tokenizer.fit_on_texts(X)
        sequences = self.tokenizer.texts_to_sequences(X)
        X = pad_sequences(sequences, maxlen=self.max_len)
        y = y.values.reshape(-1, 1)
        self.one_hot_encoder.fit(y)
        y = self.one_hot_encoder.transform(y).toarray()

        self.model = Sequential([
            Embedding(self.max_words, 128, input_length=self.max_len),
            Bidirectional(LSTM(64)),
            Dense(32, activation='relu'),
            Dense(y.shape[1], activation='sigmoid'),
        ])
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size)

    def predict_proba(self, X):
        sequences = self.tokenizer.texts_to_sequences(X)
        X = pad_sequences(sequences, maxlen=self.max_len)
        y = self.model.predict(X)
        return y
    
    def predict(self, X):
        y = self.predict_proba(X)
        y = self.one_hot_encoder.inverse_transform(y)
        return y
    
    def predict_nth_likely(self, X, n=1, threshold=0.7):
        if n < 1:
            return np.array([None] * len(X))

        y = self.predict_proba(X)

        if n > y.shape[1]:
            return np.array([None] * len(X))

        for _ in range(n - 1):
            y[0, y.argmax(axis=1)] = 0

        if y.max() > threshold:
            y = self.one_hot_encoder.inverse_transform(y)
        else:
            y = np.array([None] * len(X))

        return y
    
    def evaluate(self, X, y):
        sequences = self.tokenizer.texts_to_sequences(X)
        X = pad_sequences(sequences, maxlen=self.max_len)
        y = self.one_hot_encoder.transform(y.values.reshape(-1, 1)).toarray()
        return self.model.evaluate(X, y)
    
    def score(self, X, y):
        _, accuracy = self.evaluate(X, y)
        return accuracy
    
    def save(self, path):
        model_path = os.path.join(path, self.model_folder)
        one_hot_encoder_path = os.path.join(path, self.one_hot_encoder_file)

        os.makedirs(path, exist_ok=True)
        pickle.dump(self.one_hot_encoder, open(one_hot_encoder_path, 'wb'))
        self.model.save(model_path)

    def load(self, path):
        model_path = os.path.join(path, self.model_folder)
        one_hot_encoder_path = os.path.join(path, self.one_hot_encoder_file)

        self.one_hot_encoder = pickle.load(open(one_hot_encoder_path, 'rb'))
        self.model = load_model(model_path)
    
    