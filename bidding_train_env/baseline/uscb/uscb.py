import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from bidding_train_env.baseline.uscb.replay_buffer import ReplayBuffer
from copy import deepcopy


class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mean, sigma, theta=.15, time=1e-2, init_x=None):
        self.theta = theta
        self.mean = mean
        self.sigma = sigma
        self.time = time
        self.init_x = init_x
        self.prev_x = None
        self.reset()

    def __call__(self):
        normal = np.random.normal(size=self.mean.shape)
        new_x = self.prev_x + self.theta * (self.mean - self.prev_x) \
            * self.time + self.sigma * np.sqrt(self.time) * normal
        self.prev_x = new_x
        return new_x

    def reset(self):
        if self.init_x is not None:
            self.prev_x = self.init_x
        else:
            self.prev_x = np.zeros_like(self.mean)


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
    def __init__(self, lr=1e-3, discount_factor=1, num_action=2, epsilon=1, batch_size=100, state_dim=16):
        super().__init__()
        self.action_space = [i for i in range(num_action)]
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.step_counter = 0
        self.tau = 5e-3
        self.buffer = ReplayBuffer(100000, state_dim)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.actor = Actor(state_dim, num_action, 100, 100).to(self.device)
        self.actor_target = deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr)
        self.critic = Critic(state_dim, num_action, 100, 100).to(self.device)
        self.critic_target = deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr)
        # self.model_path = model_path
        self.num_action = num_action
        self.exploration_sigma = 0.2
        self.action_lbs = np.array([0.5, 0])
        self.action_ubs = np.array([17, 1])


    def store_tuple(self, state, action, reward):
        self.buffer.store_tuples(state, action, reward)


    def policy(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        dist_1 = np.random.normal(0, 0.1, size=state.shape[0]).reshape(-1, 1)
        dist_2 = np.random.normal(0, 0.5, size=state.shape[0]).reshape(-1, 1)
        dist = np.concatenate([dist_1,dist_2], axis=1)
        action = self.actor(state).cpu().data.numpy() + dist
        return action.flatten()


    def update(self):
        if self.buffer.counter < self.batch_size:
            return

        state_batch, action_batch, reward_batch, new_state_batch, done_batch = \
            self.buffer.sample_buffer(self.batch_size)

        # np array to tensor
        state_batch = torch.tensor(state_batch).float().to(self.device)
        action_batch = torch.tensor(
            action_batch).float().reshape(-1, 1).to(self.device)
        reward_batch = torch.tensor(
            reward_batch).float().reshape(-1, 1).to(self.device)
        new_state_batch = torch.tensor(new_state_batch).float().to(self.device)
        done_batch = torch.tensor(
            done_batch).float().reshape(-1, 1).to(self.device)

        # Compute the target Q value
        target_q = self.critic_target(
            new_state_batch, self.actor_target(new_state_batch))
        target_q = reward_batch + \
            ((1-done_batch) * self.discount_factor * target_q).detach()

        # Get current Q estimate
        current_q = self.critic(state_batch, action_batch)
        # Compute critic loss
        critic_loss = F.mse_loss(current_q, target_q)
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Compute actor loss
        actor_loss = -self.critic(state_batch, self.actor(state_batch)).mean()
        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update the frozen target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data)

        self.step_counter += 1


    def save_jit(self, save_path):
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        jit_model = torch.jit.script(self.cpu())
        torch.jit.save(jit_model, f'{save_path}/uscb_model.pth')


    # def save_model(self):
    #     # 保存模型
    #     if not os.path.exists(self.model_path):
    #         # 如果文件夹不存在，则创建一个新的文件夹
    #         os.makedirs(self.model_path)
    #     model_save_path = self.model_path+'/'+self.env_name + '_actor.pth'
    #     torch.save(self.actor.state_dict(), model_save_path)
    #     model_save_path = self.model_path+'/'+self.env_name + '_critic.pth'
    #     torch.save(self.critic.state_dict(), model_save_path)
    #     # print(f'Model saved to {model_save_path}')

    # def load_model(self):
    #     # 加载模型
    #     model_save_path = self.model_path+'/'+self.env_name + '_actor.pth'
    #     self.actor.load_state_dict(torch.load(model_save_path))
    #     self.actor.eval()
    #     model_save_path = self.model_path+'/'+self.env_name + '_critic.pth'
    #     torch.save(self.critic.state_dict(), model_save_path)
    #     self.critic.eval()

