import os
from argparse import ArgumentParser
from app import run_app
from jobs.ml_jobs import tune_legalpha, test_legalpha, train_legalpha
from utils.random_seed import RANDOM_SEED, set_random_seeds

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--tune', action='store_true', help='Tune the hyperparameters of the model')
    parser.add_argument('--tune-iters', type=int, default=10, help='Number of iterations for hyperparameter tuning')
    parser.add_argument('--tune-cv-folds', type=int, default=5, help='Number of cross validation folds')
    parser.add_argument('--test', action='store_true', help='Test the model')
    parser.add_argument('--test-size', type=float, default=0.2, help='Ratio of test data')
    parser.add_argument('--train', action='store_true', help='Train and save the model (Legalpha BERT Classifier)')
    parser.add_argument('--model-name', type=str, default='bert-classifier', help='Model to test')
    args = parser.parse_args()

    set_random_seeds(RANDOM_SEED)

    if args.tune:
        tune_legalpha(model_name=args.model_name, n_iter=args.tune_iters, cv=args.tune_cv_folds, test_size=args.test_size, random_state=RANDOM_SEED)
    elif args.test:
        test_legalpha(model_name=args.model_name, test_size=args.test_size, random_state=RANDOM_SEED)
    elif args.train:
        train_legalpha()
    else:
        run_app(random_seed=RANDOM_SEED)