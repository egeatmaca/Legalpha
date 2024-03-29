import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
# from skopt import BayesSearchCV
from time import time
from utils.data import get_data
from utils.ml import MODEL_CONSTRUCTORS, HYPERPARAM_DISTRIBUTIONS, BERT_BASED_MODELS, precalculate_embeddings, read_embeddings
import os
import json


def tune_legalpha(model_name='bert-embedding-classifier', n_iter=10, cv=5, test_size=0.2, random_state=42):
    if model_name not in MODEL_CONSTRUCTORS.keys():
        raise ValueError(f'Invalid model name: {model_name}')
    
    if model_name not in HYPERPARAM_DISTRIBUTIONS.keys():
        raise ValueError(f'Model {model_name} is not supported for hyperparameter tuning')

    # Initialize the model
    model = MODEL_CONSTRUCTORS[model_name]()

    # Initialize X and y
    questions, _ = get_data()
    X = questions['text']
    y = questions['answer_id']
    
    # Precalculate embeddings for BERT Embedding Classifier
    if model_name in BERT_BASED_MODELS:
        embeddings_path = os.path.join('data', 'embeddings', f'{model_name}.json')
        if not os.path.exists(embeddings_path):
            print('Precalculating embeddings...')
            embedding_start = time()
            precalculate_embeddings(model_name=model_name, questions=questions)
            embedding_end = time()
            print(f'Embedding Time: {embedding_end - embedding_start} seconds')

        print('Reading embeddings...')
        X, y = read_embeddings(model_name=model_name)
        model.embeddings_precalculated = True

    # Split the data into train and test sets
    X_train, _, y_train, _ = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Initialize the hyperparameter search
    hyperparam_distribution = HYPERPARAM_DISTRIBUTIONS[model_name]
    randomized_search = RandomizedSearchCV(
        model,
        hyperparam_distribution,
        n_iter=n_iter,
        cv=cv,
        random_state=random_state,
        n_jobs=1
    )

    # Run the hyperparameter search
    model_name = model_name.replace('-', ' ').title()
    print(f'Searching the best hyperparameters for {model_name}...')
    search_start = time()
    randomized_search.fit(X_train, y_train)
    search_end = time()
    print(f'Search Time: {search_end - search_start} seconds')

    results_columns = ['params', 'mean_test_score', 'std_test_score', 'mean_fit_time', 'std_fit_time']
    results = pd.DataFrame(randomized_search.cv_results_)
    results = results.loc[:, results_columns] \
                    .sort_values(by='mean_test_score', ascending=False) \
                    .reset_index(drop=True)


    # Print the results
    print(f'\nBest Hyperparameters for {model_name}:', randomized_search.best_params_, sep='\n')
    print('Best Score:', randomized_search.best_score_)

    results_folder = os.path.join('ml_models', 'results', 'model_tuning')
    results_file = model_name.replace(' ', '_').lower() + '.csv'
    results_file = os.path.join(results_folder, results_file)
    os.makedirs(results_folder, exist_ok=True)
    results.to_csv(results_file, index=False)
    





