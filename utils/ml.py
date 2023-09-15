from ml_models import Legalpha
from ml_models.experiments import LegalphaClf, LegalphaSemSearch

MODEL_CONSTRUCTORS = {
    'bert-embedding-classifier': Legalpha,
    'embedding-classifier': LegalphaClf,
    'semantic-search': LegalphaSemSearch
}

HYPERPARAM_DISTRIBUTIONS = {
    'bert-embedding-classifier': {
        'hidden_layer_sizes': [[64], [64, 8], [64, 16], [64, 32], [96], [96, 8], [96, 16], [96, 32]],
        'hidden_activation': ['relu', 'leaky_relu',],
        'output_activation': ['sigmoid', 'softmax'],
        'optimizer': ['adam', 'sgd', 'rmsprop'],
        'optimizer_learning_rate': [0.01, 0.001, 0.0001],
        'loss': ['categorical_crossentropy'],
        'epochs': [100, 150, 200],
        'batch_size': [128]
    }
}

def augment_text_data(texts):
    texts_augmented = []

    for text in texts:
        words = text.split(' ')
        n_words = len(words)
        for i in range(n_words):
            for j in range(i+1, n_words):
                texts_augmented.append(' '.join(words[i:j]))

    return texts_augmented