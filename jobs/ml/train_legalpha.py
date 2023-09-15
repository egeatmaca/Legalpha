import os
from time import time
from ml_models import Legalpha
from utils.data import read_csv

def train_legalpha(pretrained_path=os.environ.get('LEGALPHA_PRETRAINED_PATH')):
    questions = read_csv('./data/questions.csv')
    answer_ids = questions['answer_id'].unique()

    X_train = questions['text']
    y_train = questions['answer_id']

    print(f'#Training Questions: {questions.shape[0]}')
    print(f'#Answers: {answer_ids.shape[0]}')

    model = Legalpha()

    print(f'Training Legalpha...')
    training_start = time()

    model.fit(X_train, y_train, validation_split=None)
    
    training_end = time()
    print(f'Training Time: {training_end - training_start} seconds')

    print('Saving Legalpha...')
    model.save(pretrained_path)

if __name__ == '__main__':
    train_legalpha()