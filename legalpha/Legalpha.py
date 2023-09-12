import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from simpletransformers.language_representation import RepresentationModel
from sklearn.preprocessing import OneHotEncoder
import pickle
import os

class Legalpha:
    bert = RepresentationModel('bert', 'bert-base-uncased', use_cuda=False)
    model_folder = 'model'
    one_hot_encoder_file = 'one_hot_encoder.pkl'

    def __init__(self):
        self.one_hot_encoder = OneHotEncoder()
        self.model = Sequential([
            Dense(64, activation='relu', input_shape=(768,)),
            Dense(10, activation='sigmoid')
        ])
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def fit(self, X, y, epochs=100, batch_size=128, validation_split=None):
        X = self.bert.encode_sentences(X, combine_strategy='mean')
        y = y.values.reshape(-1, 1)
        self.one_hot_encoder.fit(y)
        y = self.one_hot_encoder.transform(y).toarray()
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=validation_split)
    
    def predict_proba(self, X):
        X = self.bert.encode_sentences(X, combine_strategy='mean')
        y = self.model.predict(X)
        return y
    
    def predict(self, X):
        y = self.predict_proba(X)
        y = self.one_hot_encoder.inverse_transform(y)
        return y
    
    def predict_nth_likely(self, X, n=1, threshold=0.75):
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
        X = self.bert.encode_sentences(X, combine_strategy='mean')
        y = self.one_hot_encoder.transform(y.values.reshape(-1, 1)).toarray()
        return self.model.evaluate(X, y)
    
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


