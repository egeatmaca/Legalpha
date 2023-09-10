import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Bidirectional, AveragePooling1D, Flatten
from simpletransformers.language_representation import RepresentationModel
from sklearn.preprocessing import OneHotEncoder


class LegalphaBertClf:
    bert = RepresentationModel('bert', 'bert-base-uncased', use_cuda=False)

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
    
    def predict_nth_likely(self, X, n=1):
        if n < 1:
            return np.array([None] * len(X))

        y = self.predict_proba(X)

        if n > y.shape[1]:
            return np.array([None] * len(X))

        for _ in range(n - 1):
            y[0, y.argmax(axis=1)] = 0
        y = self.one_hot_encoder.inverse_transform(y)
        return y
    
    def evaluate(self, X, y):
        X = self.bert.encode_sentences(X, combine_strategy='mean')
        y = self.one_hot_encoder.transform(y.values.reshape(-1, 1)).toarray()
        return self.model.evaluate(X, y)


