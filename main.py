import os
from argparse import ArgumentParser
from app import run_app
from jobs.ml import tune_legalpha, test_legalpha, train_legalpha
from jobs.db import export_user_questions
from utils.random_seed import RANDOM_SEED, set_random_seeds

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--tune', action='store_true', help='Tune the hyperparameters of the model')
    parser.add_argument('--tune-iters', type=int, default=10, help='Number of iterations for hyperparameter tuning')
    parser.add_argument('--tune-cv-folds', type=int, default=5, help='Number of cross validation folds')
    parser.add_argument('--test', action='store_true', help='Test the model')
    parser.add_argument('--test-size', type=float, default=0.2, help='Ratio of test data')
    parser.add_argument('--test-sampling', type=str, default='random', help='Sampling method for test data (random or stratified)')
    parser.add_argument('--train', action='store_true', help='Train and save the model (Legalpha BERT Classifier)')
    parser.add_argument('--model-name', type=str, default='bert-embedding-classifier', help='Model to test')
    parser.add_argument('--export-questions', action='store_true', help='Export user questions')
    args = parser.parse_args()

    set_random_seeds(RANDOM_SEED)

    if args.tune:
        tune_legalpha(model_name=args.model_name, n_iter=args.tune_iters, cv=args.tune_cv_folds, test_size=args.test_size, random_state=RANDOM_SEED)
    elif args.test:
        test_legalpha(model_name=args.model_name, test_size=args.test_size, sampling=args.test_sampling, random_state=RANDOM_SEED)
    elif args.train:
        train_legalpha()
    elif args.export_questions:
        export_user_questions()
    else:
        run_app(random_seed=RANDOM_SEED)