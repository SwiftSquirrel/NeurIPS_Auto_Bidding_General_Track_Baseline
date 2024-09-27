import glob
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
import numpy as np
from bidding_train_env.baseline.uscb.adsimenv import adSimEnv
from bidding_train_env.strategy import PlayerBiddingStrategy
import copy
from bidding_train_env.baseline.uscb.uscb import train, test



if __name__=='__main__':
    file_name = os.path.dirname(os.path.realpath(__file__))
    dir_name = os.path.dirname(file_name)
    dir_name = os.path.dirname(dir_name)
    dir_name = os.path.dirname(dir_name)
    file_folder_path = os.path.join(
        dir_name, "data", "traffic")
    data_paths = glob.glob(os.path.join(file_folder_path, '*.csv'))
    # train data paths
    train_data_paths = [data_path for data_path in data_paths if '27' not in data_path]
    # test data paths
    test_data_paths = [
        data_path for data_path in data_paths if '27' in data_path]
    adverNums = [i for i in range(48)]

    env = adSimEnv(data_paths, adverNums)
    agent = PlayerBiddingStrategy(test=False)

    num_epoch = 10
    train(env, agent, num_epoch)
    avg_ext_score = test(env, agent, num_epoch)
    agent.mode.save('saved_model/USCBtest')


