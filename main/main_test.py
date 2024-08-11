import numpy as np
import torch
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run.run_evaluate import run_test

torch.manual_seed(1)
np.random.seed(1)

if __name__ == "__main__":
    kp_list = [5e-1, 1e-1, 5e-2, 1e-2, 5e-3, 1e-3]
    kd_list = [5e-1, 1e-1, 5e-2, 1e-2, 5e-3, 1e-3]
    ki_list = [5e-1, 1e-1, 5e-2, 1e-2, 5e-3, 1e-3]
    # kp_list = [5e-3]
    # kd_list = [5e-1]
    # ki_list = [5e-2]
    # w0: 0.005, 0.5, 0.05
    for kp in kp_list:
        for kd in kd_list:
            for ki in ki_list:
                print(f'====== kp {kp}, kd {kd}, ki {ki} ========')
                run_test(kp, kd, ki)
