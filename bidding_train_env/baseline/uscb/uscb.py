import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from bidding_train_env.baseline.uscb.replay_buffer import ReplayBuffer
from rnd_model import TargetModel, PredictorModel
import copy

def train(env, agent, num_epoch):
    for i in range(num_epoch):
        env.reset()
        while env.data_path_adverNum_product:
            # config which experiment to run (the period, adverNum, budget, cpa, category)
            env.config()
            # modify agent config
            agent.budget = env.budget
            agent.cpa = env.cpa
            agent.category = env.category
            agent.reset()

            int_R = 0
            ext_R = 0
            for i in range(len(env.timeStepIndices)):
                # current state
                state = (
                    env.timeStepIndices[i], env.pValues[i], env.pValueSigmas[i], env.history['historyPValueInfo'],
                    env.history["historyBids"], env.history["historyAuctionResult"], env.history["historyImpressionResult"],
                    env.history["historyLeastWinningCost"]
                )

            # interact with env
            bid = agent.bidding(**state, update_action=True)
            int_r_i = agent.get_int_reward(**state)
            # env history
            ext_r_i, next_state = env.step(env.timeStepIndices[i], bid, env.history)
            ext_R += ext_r_i
            int_R += int_r_i

            ext_V, int_V = 0, 0
            history = copy.deepcopy(env.history)
            # copy agent & env
            agent_tmp = copy.deepcopy(agent)
            env_tmp = copy.deepcopy(env)
            # current action
            action = (agent.w0, agent.w1, agent.r1, agent.r2)

            # ext reward, no need to consider discount factor
            # TODO: check whether int reward need to consider discount factor

            for j in range(i+1, len(env.timeStepIndicies)):
                # state update, bid calcu
                state_tmp = next_state
                bid_tmp = agent_tmp.bidding(**state_tmp, update_action=False)
                int_r_j = agent_tmp.get_int_reward(**state_tmp)
                # interact with env
                ext_r_j, next_state = env_tmp.step(
                    env_tmp.timeStepIndices[j], bid_tmp, history)
                
                ext_V += ext_r_j
                int_V += int_r_j


            # calculate kpi
            all_cost = env_tmp.budget - env_tmp.remaining_budget
            ext_score = ext_R + ext_V
            int_score = int_R + int_V

            # calculate ext_score
            cpa_real = all_cost / (ext_score + 1e-10)
            cpa_constraint = env_tmp.cpa
            ext_score = ext_score/env.optimal_conv
            penalty = 20**max(0, cpa_real/cpa_constraint-1) - 1
            ext_score - ext_score - penalty


            # int_score: need to check scale


            # add experience to buffer
            agent.model.buffer.store_tuples(state, action, ext_score, int_score)

            # update the critic and actor network
            agent.model.update()


def test(env, agent):
    ext_score_list = []
    env.reset()
    while env.data_path_adverNum_product:
        # config which experiment to run (the period, advNum, budget, cpa, category)
        env.config()
        # modify agent config
        agent.budget = env.budget
        agent.cpa = env.cpa
        agent.category = env.category
        agent.reset()

        int_R = 0
        ext_R = 0
        for i in range(len(env.timeStepIndices)):
            # current state
            state = (
                env.timeStepIndices[i], env.pValues[i], env.pValueSigmas[i], env.history["historyPValueInfo"],
                env.history["historyBids"], env.history["historyAuctionResult"], env.history["historyImpressionResult"],
                env.history["historyLeastWinningCost"]
            )

            # interact with env
            bid = agent.bidding(**state, update_action=True)
            int_r_i = agent.get_int_reward(**state)
            # env history
            ext_r_i, next_state = env.step(env.timeStepIndices[i], bid, env.history)
            ext_R += ext_r_i
            int_R += int_r_i
            state = next_state

            # calculate kpi
            all_cost = env.budget - env.remaining_budget
            ext_score = ext_R

            # claculate ext_score
            cpa_real = all_cost / (ext_score + 1e-10)
            cpa_constraint = env.cpa
            ext_score = ext_score/env.optimal_conv
            penalty = 20**max(0, cpa_real/cpa_constraint-1) - 1
            ext_score = ext_score - penalty

            ext_score_list.append(ext_score)

    return np.mean(ext_score_list)




# https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
class RunningStdMean():
    def __init__(self, shape=()):
        self.mean = np.zeros(shape, "float64")
        self.var = np.ones(shape, "float64")
        self.count = 1e-4

    def update(self, input_batch_b):
        mean_b = np.mean(input_batch_b, axis=0)
        var_b = np.var(input_batch_b, axis=0)
        count_b = input_batch_b.shape[0]
        self.var, self.mean = self.parallel_variance(self.mean, self.count,
                                                     self.var, mean_b, count_b,
                                                     var_b)
        self.count = count_b + self.count

    def update_from_mean_std(self, mean_b, var_b, count_b):
        self.var, self.mean = self.parallel_variance(self.mean, self.count,
                                                     self.var, mean_b, count_b,
                                                     var_b)
        self.count = count_b + self.count

    def parallel_variance(self, avg_a, count_a, var_a, avg_b, count_b, var_b):
        delta = avg_b - avg_a
        m_a = var_a * (count_a)
        m_b = var_b * (count_b)
        M2 = m_a + m_b + np.square(delta) * count_a * count_b / (
            count_a + count_b)
        new_mean = avg_a + delta * count_b / (count_a + count_b)
        new_var = M2 / (count_a + count_b)
        return new_var, new_mean



class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden1_dim, hidden2_dim, max_action=[17, 1]):
        super(Actor, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.l1 = nn.Linear(state_dim, hidden1_dim)
        self.l2 = nn.Linear(hidden1_dim, hidden2_dim)
        self.l3 = nn.Linear(hidden2_dim, action_dim)
        self.max_action = torch.tensor(max_action).to(self.device)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        # need to modify
        x = self.max_action * torch.sigmoid(self.l3(x))
        return x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden1_dim, hidden2_dim):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, hidden1_dim)
        self.l2 = nn.Linear(hidden1_dim, hidden2_dim)
        self.l3 = nn.Linear(hidden2_dim, 1)

    def forward(self, x, u):
        x = F.relu(self.l1(torch.cat([x, u], 1)))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x




class Uscb(nn.Module):
    def __init__(self, lr=1e-3, discount_factor=1, num_action=4, epsilon=1, batch_size=100, state_dim=16):
        super().__init__()
        self.action_space = [i for i in range(num_action)]
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.batch_size = batch_size
        # self.step_counter = 0
        self.ext_adv_coef = 2
        self.int_adv_coef = 1
        self.tau = 5e-3
        self.buffer = ReplayBuffer(100000, state_dim)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.actor = Actor(state_dim, num_action, 100, 100).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr)
        self.critic = Critic(state_dim, num_action, 100, 100).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr)

        self.target_model = TargetModel().to(self.device)
        self.predictor_model = PredictorModel().to(self.device)
        self.rnd_optimizer = optim.Adam(self.predictor_model.parameters(), lr)

        self.reward_rms = RunningStdMean()
        # check running OBS' shape
        self.obs_rms = RunningStdMean(shape=(1, 1, 84, 84))


        self.num_action = num_action
        self.exploration_sigma = 0.2
        self.action_lbs = np.array([0.5, 0, -1, -1])
        self.action_ubs = np.array([17, 1, 1, 1])


    def store_tuple(self, state, action, reward):
        self.buffer.store_tuples(state, action, reward)


    def get_intrinsic_rewards(self, input_observation):
        target_value = self.target_model(input_observation)
        predictor_value = self.predictor_model(
            input_observation
        )
        intrinsic_reward = (target_value - predictor_value).pow(2).sum(1)/2
        intrinsic_reward = intrinsic_reward.data.cpu().numpy()

        # use running mean and std to update the reward

        return intrinsic_reward

    def warm_up(self):
        # TODO: may not need to calculate running mean & std for obs; its okay just for reward
        # 
        # run a couple of experiments to get the observations
        observations_to_normalize = []
        self.obs_rms.update(observations_to_normalize)




    def policy(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)

        # add some noise to the action, for better exploration
        dist_1 = np.random.normal(0, 0.5, size=state.shape[0]).reshape(-1, 1)
        dist_2 = np.random.normal(0, 0.5, size=state.shape[0]).reshape(-1, 1)
        dist_3 = np.random.normal(0, 0.5, size=state.shape[0]).reshape(-1, 1)
        dist_4 = np.random.normal(0, 0.5, size=state.shape[0]).reshape(-1, 1)

        dist = np.concatenate([dist_1, dist_2, dist_3, dist_4], axis=1)

        action = self.actor(state).cpu().data.numpy() + dist

        return action.flatten()


    def update(self):
        if self.buffer.counter < self.batch_size:
            return

        state_batch, action_batch, reward_batch = \
            self.buffer.sample_buffer(self.batch_size)

        # np array to tensor
        state_batch = torch.tensor(state_batch).float().to(self.device)
        action_batch = torch.tensor(
            action_batch).float().reshape(-1, 1).to(self.device)
        reward_batch = torch.tensor(
            reward_batch).float().reshape(-1, 1).to(self.device)

        # prediction by critic
        pred_reward = self.critic(state_batch, action_batch)
        # compute critic loss
        critic_loss = F.mse_loss(reward_batch, pred_reward)
        # optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # compute actor loss
        actor_loss = -self.critic(state_batch, self.actor(state_batch)).mean()
        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # self.step_counter += 1


    def save_jit(self, save_path):
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        jit_model = torch.jit.script(self.cpu())
        torch.jit.save(jit_model, f'{save_path}/uscb_model.pth')


