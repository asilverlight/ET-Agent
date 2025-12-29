import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default=None)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--exp_type", type=str, default=None)
    parser.add_argument("--endpoints", type=str, default=None, nargs="+")
    parser.add_argument("--api_keys", type=str, default=None, nargs="+")
    parser.add_argument("--default_model", type=str, default=None)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--concurrent_limit", type=int, default=64)
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--round", type=int, default=10)
    args = parser.parse_args()
    return args