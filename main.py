from argparse import ArgumentParser
from app import run_app
from jobs.test_legalpha import test_legalpha

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--test', action='store_true', help='Test Legalpha')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test size for Legalpha')
    args = parser.parse_args()

    if args.test:
        test_legalpha(test_size=args.test_size)
    else:
        run_app()