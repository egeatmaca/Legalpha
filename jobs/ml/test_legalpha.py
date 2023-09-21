import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from time import time
import os
import json
from utils.data import get_data
from utils.ml import MODEL_CONSTRUCTORS, BERT_BASED_MODELS, EMBEDDINGS_DIR, precalculate_embeddings, read_embeddings

def stratified_train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True):
    X_train_list = []
    X_test_list = []
    y_train_list = []
    y_test_list = []
    y_unique = y.unique()
    for y_value in y_unique:
        X_of_y_value = X.loc[y == y_value]
        y_of_y_value = y.loc[y == y_value]

        if shuffle:
            X_of_y_value = X_of_y_value.sample(frac=1, random_state=random_state)

        X_test = X_of_y_value.sample(frac=test_size, random_state=random_state)
        y_test = y_of_y_value.loc[X_test.index]
        
        train_index = X_of_y_value.index.difference(X_test.index)
        X_train = X_of_y_value.loc[train_index]
        y_train = y_of_y_value.loc[train_index]
        
        X_train_list.append(X_train)
        X_test_list.append(X_test)
        y_train_list.append(y_train)
        y_test_list.append(y_test)

    X_train = pd.concat(X_train_list)
    X_test = pd.concat(X_test_list)
    y_train = pd.concat(y_train_list)
    y_test = pd.concat(y_test_list)

    return X_train, X_test, y_train, y_test

def predict_all(questions_pred, legalpha):
    answers_pred = []
    for question in questions_pred['text']:
        _, answer_id, _ = legalpha.predict(question)
        answers_pred.append(answer_id)
    return answers_pred

def evaluate(answers_true, answers_predicted):
    return classification_report(answers_true, answers_predicted)

def test_legalpha(model_name='bert-classifier', test_size=0.2, sampling='random', random_state=42):
    if model_name not in MODEL_CONSTRUCTORS.keys():
        raise ValueError(f'Invalid model name: {model_name}')

    print(f'Test Size: {test_size}')
    print(f'Sampling Method: {sampling}')
    print(f'Random State: {random_state}')

    # Precalculate embeddings for BERT Embedding Classifier
    embeddings_precalculated = False
    if model_name in BERT_BASED_MODELS:
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

        embeddings_precalculated = True
    else:
        print('Reading data...')
        questions, answers = get_data()
        X = questions['text']
        y = questions['answer_id']

    if sampling == 'random':
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    elif sampling == 'stratified':
        X_train, X_test, y_train, y_test = stratified_train_test_split(X, y, test_size=test_size, random_state=random_state)
    else:
        raise ValueError(f'Invalid sampling method: {sampling}')

    print(f'#Training Samples: {X_train.shape[0]}')
    print(f'#Test Samples: {X_test.shape[0]}')
    print(f'#Labels: {y.nunique()}')

    model = MODEL_CONSTRUCTORS[model_name]() \
            if not embeddings_precalculated \
            else MODEL_CONSTRUCTORS[model_name](embeddings_precalculated=True)

    print(f'Training {model_name}...')
    training_start = time()

    if model_name == 'semantic-search':
        questions_train = pd.concat([X_train, y_train], axis=1)
        model.fit(questions_train, answers)
    else:
        model.fit(X_train, y_train)
    
    training_end = time()
    print(f'Training Time: {training_end - training_start} seconds')

    print(f'Testing {model_name}...')
    test_start = time()
    
    y_pred = None
    if model_name == 'semantic-search':
        questions_test = pd.concat([X_test, y_test], axis=1)
        y_pred = predict_all(questions_test, model)
    else:
        y_pred = model.predict(X_test)
    test_results = evaluate(y_test, y_pred)

    test_end = time()
    print(f'Test Time: {test_end - test_start} seconds')
    print(f'Test Results: \n{test_results}')

if __name__ == '__main__':
    test_legalpha()