from argparse import ArgumentParser
from app import run_app
from jobs.test_legalpha import test_legalpha, test_legalpha_clf

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--test', type=str, default='semantic-search', help='Model to test')
    parser.add_argument('--test_size', type=float, default=0.2, help='Ratio of test samples to total samples')
    args = parser.parse_args()

    if args.test == 'semantic-search':
        test_legalpha(test_size=args.test_size)
    elif args.test == 'classifier':
        test_legalpha_clf(test_size=args.test_size)
    elif args.test == 'bert-classifier':
        test_legalpha_clf(test_size=args.test_size, bert_embeddings=True)
    else:
        run_app()