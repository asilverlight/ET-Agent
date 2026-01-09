import sys
import os

sys.path.append(os.getcwd())

import json
from typing import List, Tuple

from .utils import extract_solution, last_boxed_only_string, remove_boxed


class DataLoader:
    """Loading Datasets"""

    def __init__(self, args):
        self.data_path = args.data_path
        self.counts = args.counts

    def load_data(self):
        with open(self.data_path, 'r', encoding='utf-8') as f:
            datas = json.load(f)
            # print(self.counts)
            datas = datas[:self.counts]
            return datas

    def load_pool(self):
        with open(self.data_path, 'r', encoding='utf-8') as f:
            datas = json.load(f)
            return datas