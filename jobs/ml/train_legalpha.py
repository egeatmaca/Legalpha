import os
from time import time
from ml_models import Legalpha
from utils.data import get_data
from utils.ml import EMBEDDINGS_DIR, precalculate_embeddings, read_embeddings

def train_legalpha(pretrained_path=os.environ.get('LEGALPHA_PRETRAINED_PATH')):
    model_name = 'bert-classifier'
    embeddings_file = model_name.replace('-', '_') + '.json'
    embeddings_path = os.path.join(EMBEDDINGS_DIR, embeddings_file)

    if not os.path.exists(embeddings_path):
        print('Precalculating embeddings...')
        embedding_start = time()
        precalculate_embeddings(model_name=model_name)
        embedding_end = time()
        print(f'Embedding Time: {embedding_end - embedding_start} seconds')

    print('Reading embeddings...')
    X, y = read_embeddings(model_name=model_name)

    print(f'#Training Samples: {X.shape[0]}')
    print(f'#Labels: {y.nunique()}')

    model = Legalpha()
    model.embeddings_precalculated = True

    print(f'Training Legalpha...')
    training_start = time()

    model.fit(X, y, validation_split=None)
    
    training_end = time()
    print(f'Training Time: {training_end - training_start} seconds')

    print('Saving Legalpha...')
    model.save(pretrained_path)

if __name__ == '__main__':
    train_legalpha()