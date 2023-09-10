from argparse import ArgumentParser
from app import run_app
from jobs.test_legalpha import test_legalpha
from jobs.train_legalpha import train_legalpha

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--test', action='store_true', help='Test mode')
    parser.add_argument('--test-model', type=str, default='bert-classifier', help='Model to test')
    parser.add_argument('--test-size', type=float, default=0.2, help='Ratio of test samples to total samples')
    parser.add_argument('--train', action='store_true', help='Train the model (Legalpha Bert Classifier)')
    args = parser.parse_args()

    if args.test:
        test_legalpha(model_name=args.test_model, test_size=args.test_size)
    elif args.train:
        train_legalpha()
    else:
        run_app()