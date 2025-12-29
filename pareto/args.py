import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default=None)
    parser.add_argument("--data_path2", default=None)
    parser.add_argument("--data_path3", default=None)
    parser.add_argument("--still_path", default=None)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--exp_type", type=str, default=None)
    parser.add_argument("--counts", type=int, default=None)
    args = parser.parse_args()
    return args