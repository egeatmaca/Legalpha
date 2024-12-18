import numpy as np
import pandas as pd
import os
import json
from nlpaug.augmenter.word.synonym import SynonymAug
from ml_models import Legalpha
from ml_models.experiments import LegalphaEmbedLSTMClf, LegalphaSemSearch, LegalphaBertLSTMClf
from utils.data import get_data

MODEL_CONSTRUCTORS = {
    'bert-classifier': Legalpha,
    'bert-lstm-classifier': LegalphaBertLSTMClf,
    'embedding-lstm-classifier': LegalphaEmbedLSTMClf,
    'semantic-search': LegalphaSemSearch
}

HYPERPARAM_DISTRIBUTIONS = {
    'bert-classifier': {
        'hidden_layer_sizes': [[64], [96], [128], [64, 16], [64, 32], [96, 16], [96, 32], [128, 16], [128, 32]],
        'hidden_activation': ['relu', 'leaky_relu',],
        'output_activation': ['sigmoid', 'softmax'],
        'optimizer': ['adam', 'sgd', 'rmsprop'],
        'optimizer_learning_rate': [0.01, 0.001, 0.0001],
        'layer_normalization': [True, False],
    },
    'bert-lstm-classifier': {
        'hidden_layer_sizes': [[16, 16], [32, 16], [32, 32], [64, 32]],  
        'hidden_activation': ['relu', 'leaky_relu',],
        'output_activation': ['sigmoid', 'softmax'],
        'optimizer': ['adam', 'sgd', 'rmsprop'],
        'optimizer_learning_rate': [0.01, 0.001, 0.0001],
        'layer_normalization': [True, False],
    },
}

BERT_BASED_MODELS = ['bert-classifier', 'bert-lstm-classifier']

EMBEDDINGS_DIR = os.path.join('data', 'embeddings')


def precalculate_embeddings(model_name='bert-classifier', questions=None):
    if model_name not in BERT_BASED_MODELS:
        raise ValueError('Embeddings can only be precalculated for BERT-based models.')
    
    if questions is None:
        questions, _ = get_data()
    
    model = MODEL_CONSTRUCTORS[model_name]()
    
    embeddings_dict = {
        'embeddings': model.encode_sentences(questions['text']).tolist(),
        'labels': questions['answer_id'].tolist()
    }

    embeddings_file = model_name.replace('-', '_') + '.json'
    embeddings_path = os.path.join(EMBEDDINGS_DIR, embeddings_file)
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
    json.dump(embeddings_dict, open(embeddings_path, 'w'))


def read_embeddings(model_name='bert-classifier'):
    embeddings_file = model_name.replace('-', '_') + '.json'
    embeddings_path = os.path.join(EMBEDDINGS_DIR, embeddings_file)
    embeddings_dict = json.load(open(embeddings_path, 'r'))
    embeddings = pd.DataFrame(embeddings_dict['embeddings'])
    labels = pd.Series(embeddings_dict['labels'])
    return embeddings, labels


def augment_data(X, y, n=5):
        aug = SynonymAug()
        X, y = np.array(X), np.array(y)
        X_aug, y_aug = list(X), list(y)
        for label in np.unique(y):
            X_of_label = X[y == label]
            X_aug_of_label = pd.Series(X_of_label).apply(lambda x: aug.augment(x, n=n)).explode()
            X_aug.extend(X_aug_of_label)
            y_aug.extend([label] * len(X_aug_of_label))
        
        return X_aug, y_aug