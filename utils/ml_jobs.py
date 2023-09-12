import pandas as pd
from ml_models import Legalpha
from ml_models.experiments import LegalphaClf, LegalphaSemSearch

MODEL_CONSTRUCTORS = {
    'bert-embedding-classifier': Legalpha,
    'embedding-classifier': LegalphaClf,
    'semantic-search': LegalphaSemSearch
}

HYPERPARAM_DISTRIBUTIONS = {
    'bert-embedding-classifier': {
        'hidden_layer_sizes': [
            [16], [32], [64], [128], # 1. Iteration
        ],
        'hidden_activation': ['relu', 'leaky_relu', 'elu', 'selu', 'swish'],
        'output_activation': ['sigmoid', 'softmax'],
        'optimizer': ['adam', 'sgd', 'rmsprop', 'adagrad'],
        'optimizer_learning_rate': [0.01, 0.001, 0.0001],
        'loss': ['categorical_crossentropy'],
        'epochs': [10, 50, 100, 200],
        'batch_size': [64, 128, 256]
    }
}

def get_data():
    questions = pd.read_csv('./data/questions.csv')
    answers = pd.read_csv('./data/answers.csv')
    return questions, answers