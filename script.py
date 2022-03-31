import os
import sys
import shutil
curr_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(str(Path(curr_path).parent))
import argparse
from pathlib import Path
from tqdm import tqdm
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
from natsort import natsorted

import pdb
st = pdb.set_trace


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, default=None)
    parser.add_argument('--dst', type=str, default=None)
    args = parser.parse_args()

    os.makedirs(args.dst, exist_ok=True)
    exit(0)