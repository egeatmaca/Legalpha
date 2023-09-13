import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from time import time
from utils.ml_jobs import MODEL_CONSTRUCTORS, HYPERPARAM_DISTRIBUTIONS, get_data


def tune_legalpha(model_name='bert-embedding-classifier', n_iter=10, cv=5, test_size=0.2, random_state=42):
    if model_name not in MODEL_CONSTRUCTORS.keys():
        raise ValueError(f'Invalid model name: {model_name}')
    
    if model_name not in HYPERPARAM_DISTRIBUTIONS.keys():
        raise ValueError(f'Model {model_name} is not supported for hyperparameter tuning')

    questions, _ = get_data()
    X = questions['text']
    y = questions['answer_id']
    X_train, _, y_train, _ = train_test_split(X, y, test_size=test_size, random_state=random_state)

    model = MODEL_CONSTRUCTORS[model_name]()
    hyperparam_distribution = HYPERPARAM_DISTRIBUTIONS[model_name]
    randomized_search = RandomizedSearchCV(
        model,
        hyperparam_distribution,
        n_iter=n_iter,
        cv=cv,
        random_state=random_state,
        n_jobs=1
    )

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

    print(f'\nBest Hyperparameters for {model_name}:', randomized_search.best_params_, sep='\n')
    print('Best Score:', randomized_search.best_score_)
    
    print('\n', 'All Results:')
    for i, row in results.iterrows():
        print(f'#{i + 1}:', row.to_dict())





