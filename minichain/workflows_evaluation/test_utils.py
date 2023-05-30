import argparse


def get_test_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--interact",
        "-i",
        action="store_true",
        help="if run interactively",
    )
    args = parser.parse_args()
    return args
