from argparse import ArgumentParser
from app import run_app
from jobs.test_legalpha import test_legalpha

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--test', action='store_true', help='Test the model')
    parser.add_argument('--model-name', type=str, default='bert-classifier', help='Model to test')
    parser.add_argument('--test-size', type=float, default=0.2, help='Ratio of test samples to total samples')
    args = parser.parse_args()

    if args.test:
        test_legalpha(model_name=args.model_name, test_size=args.test_size)
    else:
        run_app()