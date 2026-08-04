"""
seed_utils.py
Locks random seeds across Python's `random` module, NumPy, and PyTorch
(CPU + CUDA) for reproducibility
"""

import os
import random
import numpy as np
import torch

# Fix all relevant RNGs to 'seed'
def set_seed(seed: int = 42, deterministic: bool = True) -> None:

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
