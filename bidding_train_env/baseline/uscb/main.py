# from agent import Agent
# import gymnasium as gym

# env_name = 'MountainCarContinuous-v0'
# env = gym.make(env_name)
# state_dim = env.observation_space.shape[0]
# num_action = env.action_space.shape[0]
# train = False
# num_episodes = 50
# model_path = './models'
# device = 'cuda:0'

# dqn_agent = Agent(lr=1e-3, discount_factor=0.999, num_action=num_action,
#                   epsilon=1.0, batch_size=256, state_dim=state_dim, env_name=env_name, model_path=model_path, device=device)

# if train:
#     dqn_agent.train_model(env, num_episodes)
# else:
#     score = dqn_agent.test_model(render_mode='human', load_model=True)
#     print(score)



import glob
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
import numpy as np
import torch
import bidding_train_env
from bidding_train_env.baseline.uscb.adsimulationenv import adSimulationEnv
from bidding_train_env.strategy import PlayerBiddingStrategy



if __name__ == '__main__':
    file_name = os.path.dirname(os.path.realpath(__file__))
    dir_name = os.path.dirname(file_name)
    dir_name = os.path.dirname(dir_name)
    dir_name = os.path.dirname(dir_name)
    file_folder_path = os.path.join(
        dir_name, "data", "traffic")
    data_paths = glob.glob(os.path.join(file_folder_path, '*.csv'))
    adverNums = [i for i in range(48)]
    env = adSimulationEnv(data_paths, adverNums)
    agent = PlayerBiddingStrategy(test=False)
    num_epoch = 2

    for i in range(num_epoch):
        env.reset()
        while env.data_path_adverNum_product:
            env.run(agent)

    agent.model.save_jit('saved_model/USCBtest')

