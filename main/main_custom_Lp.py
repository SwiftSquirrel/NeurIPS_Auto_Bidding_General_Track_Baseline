import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from run.run_custom_Lp import run_custom_Lp
import numpy as np
import torch



torch.manual_seed(1)
np.random.seed(1)

if __name__ == "__main__":
    run_custom_Lp()
