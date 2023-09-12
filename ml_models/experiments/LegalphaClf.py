from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Bidirectional, AveragePooling1D, Flatten
from sklearn.preprocessing import OneHotEncoder


class LegalphaClf:
    def __init__(self, max_words=10000, max_len=500):
        self.max_words = max_words
        self.max_len = max_len
        self.tokenizer = Tokenizer(num_words=self.max_words)
        self.one_hot_encoder = OneHotEncoder()
        self.model = Sequential([
            Embedding(self.max_words, 64, input_length=self.max_len),
            Flatten(),
            Dense(32, activation='relu'),
            Dense(10, activation='sigmoid'),
        ])
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def fit(self, X, y, epochs=100, batch_size=128, validation_split=None):
        self.tokenizer.fit_on_texts(X)
        sequences = self.tokenizer.texts_to_sequences(X)
        X = pad_sequences(sequences, maxlen=self.max_len)
        y = y.values.reshape(-1, 1)
        self.one_hot_encoder.fit(y)
        y = self.one_hot_encoder.transform(y).toarray()
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=validation_split)

    def predict(self, X):
        sequences = self.tokenizer.texts_to_sequences(X)
        X = pad_sequences(sequences, maxlen=self.max_len)
        y = self.model.predict(X)
        y = self.one_hot_encoder.inverse_transform(y)
        return y
    
    def evaluate(self, X, y):
        sequences = self.tokenizer.texts_to_sequences(X)
        X = pad_sequences(sequences, maxlen=self.max_len)
        y = self.one_hot_encoder.transform(y.values.reshape(-1, 1)).toarray()
        return self.model.evaluate(X, y)


