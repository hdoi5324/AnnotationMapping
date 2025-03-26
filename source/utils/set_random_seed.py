import sys
import random
import numpy as np

def set_random_seed(seed):
    """"Attempts to lock things down to make it repeatable for the same seed"""
    random.seed(seed)
    np.random.seed(seed)

    if 'torch' in sys.modules:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
