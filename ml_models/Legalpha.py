import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import OneHotEncoder
from simpletransformers.language_representation import RepresentationModel
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LayerNormalization
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.optimizers.experimental import Nadam
import pickle
import os

class Legalpha(BaseEstimator, ClassifierMixin):
    bert = RepresentationModel('bert', 'bert-base-uncased', use_cuda=False)
    optimizers = { 'adam': Adam, 'nadam': Nadam, 'sgd': SGD, 'rmsprop': RMSprop }
    model_folder = 'model'
    one_hot_encoder_file = 'one_hot_encoder.pkl'

    def __init__(self, hidden_layer_sizes=[128], 
                 hidden_activation='leaky_relu', output_activation='softmax', 
                 optimizer='rmsprop', optimizer_learning_rate=0.001,
                 loss='categorical_crossentropy', metrics=['accuracy'],
                 layer_normalization=False,
                 embeddings_precalculated=False):
        if metrics is None:
            metrics = ['accuracy']
        elif 'accuracy' not in metrics:
            metrics = ['accuracy'] + metrics
        elif metrics[0] != 'accuracy':
            metrics.remove('accuracy')
            metrics = ['accuracy'] + metrics

        self.hidden_layer_sizes = hidden_layer_sizes
        self.hidden_activation = hidden_activation
        self.output_layer_size = None
        self.output_activation = output_activation
        self.optimizer = optimizer
        self.optimizer_learning_rate = optimizer_learning_rate
        self.loss = loss
        self.metrics = metrics
        self.layer_normalization = layer_normalization
        self.embeddings_precalculated = embeddings_precalculated

    def encode_sentences(self, X):
        return self.bert.encode_sentences(X, combine_strategy='mean')

    def generate_model(self):
        model = Sequential()
        
        for i, layer_size in enumerate(self.hidden_layer_sizes):
            if self.layer_normalization:
                model.add(LayerNormalization())

            if i == 0:
                model.add(Dense(layer_size, activation=self.hidden_activation, input_shape=(768,)))
            else:
                model.add(Dense(layer_size, activation=self.hidden_activation))

        model.add(Dense(self.output_layer_size, activation='sigmoid'))

        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)

        return model
        
    def fit(self, X, y, batch_size=128, epochs=150, validation_split=None):
        if not self.embeddings_precalculated:
            X = self.bert.encode_sentences(X, combine_strategy='mean')
        
        y = np.array(y).reshape(-1, 1)
        self.one_hot_encoder = OneHotEncoder()
        self.one_hot_encoder.fit(y)
        y = self.one_hot_encoder.transform(y).toarray()
        self.output_layer_size = y.shape[1]
        
        self.model = self.generate_model()
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=validation_split)
    
    def predict_proba(self, X):
        if not self.embeddings_precalculated:
            X = self.bert.encode_sentences(X, combine_strategy='mean')
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
        if not self.embeddings_precalculated:
            X = self.bert.encode_sentences(X, combine_strategy='mean')
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


