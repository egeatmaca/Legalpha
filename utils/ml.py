from ml_models import Legalpha
from ml_models.experiments import LegalphaClf, LegalphaSemSearch

MODEL_CONSTRUCTORS = {
    'bert-embedding-classifier': Legalpha,
    'legalpha-classifier': LegalphaClf,
    'semantic-search': LegalphaSemSearch
}

HYPERPARAM_DISTRIBUTIONS = {
    'bert-embedding-classifier': {
        'hidden_layer_sizes': [[64], [96], [128], [64, 16], [64, 32], [96, 16], [96, 32], [128, 16], [128, 32]],
        'hidden_activation': ['relu', 'leaky_relu',],
        'output_activation': ['sigmoid', 'softmax'],
        'optimizer': ['adam', 'sgd', 'rmsprop'],
        'optimizer_learning_rate': [0.01, 0.001, 0.0001],
        'layer_normalization': [True, False],
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