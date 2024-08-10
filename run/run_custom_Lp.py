import numpy as np
import logging
from bidding_train_env.common.utils import normalize_state, normalize_reward, save_normalize_dict
from bidding_train_env.baseline.iql.replay_buffer import ReplayBuffer
from bidding_train_env.baseline.custom_Lp.custom_Lp import Custom_Lp

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(name)s] [%(filename)s(%(lineno)d)] [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def run_custom_Lp():
    """
    Run customlp model training and evaluation.
    """
    customlp = Custom_Lp("./data/traffic/")
    customlp.train("saved_model/customLpTest")


if __name__ == '__main__':
    run_custom_Lp()
