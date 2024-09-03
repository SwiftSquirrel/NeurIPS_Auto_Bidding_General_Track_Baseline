import random
import pandas as pd
from bidding_train_env.dataloader.test_dataloader import TestDataLoader
from bidding_train_env.environment.offline_env import OfflineEnv
import numpy as np
import math
import copy
import itertools


class adSimulationEnv:

    def __init__(self, data_paths, adverNums):
        self.data_paths = data_paths
        self.adverNums = adverNums
        self.env = OfflineEnv()
        self.optimal_convs = pd.read_csv(
            '/home/dawn/NeurIPS_Auto_Bidding_General_Track_Baseline/saved_model/customLpTest/for_obj/bidding_param_for_obj.csv')


    def reset(self):
        """ 重置环境 """
        random.shuffle(self.data_paths)
        random.shuffle(self.adverNums)
        self.data_path_adverNum_product = list(
            itertools.product(self.data_paths, self.adverNums))


    def step(self, timeStepIndex, agent, history, update_action = True):

        pValue = self.pValues[timeStepIndex]
        pValueSigma = self.pValueSigmas[timeStepIndex]
        leastWinningCost = self.leastWinningCosts[timeStepIndex]

        if agent.remaining_budget < self.env.min_remaining_budget:
            bid = np.zeros(pValue.shape[0])
        else:
            bid = agent.bidding(timeStepIndex, pValue, pValueSigma, history["historyPValueInfo"],
                                history["historyBids"],
                                history["historyAuctionResult"], history["historyImpressionResult"],
                                history["historyLeastWinningCost"], update_action=update_action)

        tick_value, tick_cost, tick_status, tick_conversion = self.env.simulate_ad_bidding(pValue, pValueSigma, bid,
                                                                                           leastWinningCost)

        # Handling over-cost (a timestep costs more than the remaining budget of the bidding advertiser)
        over_cost_ratio = max(
            (np.sum(tick_cost) - agent.remaining_budget) / (np.sum(tick_cost) + 1e-4), 0)
        while over_cost_ratio > 0:
            pv_index = np.where(tick_status == 1)[0]
            dropped_pv_index = np.random.choice(pv_index, int(math.ceil(pv_index.shape[0] * over_cost_ratio)),
                                                replace=False)
            bid[dropped_pv_index] = 0
            tick_value, tick_cost, tick_status, tick_conversion = self.env.simulate_ad_bidding(pValue, pValueSigma, bid,
                                                                                               leastWinningCost)
            over_cost_ratio = max(
                (np.sum(tick_cost) - agent.remaining_budget) / (np.sum(tick_cost) + 1e-4), 0)

        agent.remaining_budget -= np.sum(tick_cost)
        temHistoryPValueInfo = [(pValue[i], pValueSigma[i])
                                for i in range(pValue.shape[0])]
        history["historyPValueInfo"].append(np.array(temHistoryPValueInfo))
        history["historyBids"].append(bid)
        history["historyLeastWinningCost"].append(leastWinningCost)
        temAuctionResult = np.array(
            [(tick_status[i], tick_status[i], tick_cost[i]) for i in range(tick_status.shape[0])])
        history["historyAuctionResult"].append(temAuctionResult)
        temImpressionResult = np.array(
            [(tick_conversion[i], tick_conversion[i]) for i in range(pValue.shape[0])])
        history["historyImpressionResult"].append(temImpressionResult)

        return np.sum(tick_conversion)

    def config(self):
        # config simulation
        data_path, adverNum = self.data_path_adverNum_product.pop()
        data_loader = TestDataLoader(file_path=data_path)
        keys, test_dict = data_loader.keys, data_loader.test_dict
        key = keys[adverNum]
        period = key[0]

        self.budget = test_dict[key]['budget'].values[0]
        self.category = test_dict[key]['advertiserCategoryIndex'].values[0]
        self.cpa = test_dict[key]['CPAConstraint'].values[0]

        self.optimal_conv = self.optimal_convs[self.optimal_convs.deliveryPeriodIndex ==
                                               period]['obj_value'].values[0]

        num_timeStepIndex, pValues, pValueSigmas, leastWinningCosts = data_loader.mock_data(
            key)
        self.num_timeStepIndex = num_timeStepIndex
        self.pValues = pValues
        self.pValueSigmas = pValueSigmas
        self.leastWinningCosts = leastWinningCosts

        self.rewards = np.zeros(num_timeStepIndex)
        self.history = {
            'historyBids': [],
            'historyAuctionResult': [],
            'historyImpressionResult': [],
            'historyLeastWinningCost': [],
            'historyPValueInfo': []
        }
        self.timeStepIndices = list(range(self.num_timeStepIndex))

        return


    def run(self, agent):
        # config which experiment to run (the period, advNum, budget, cpa, category)
        self.config()
        # modify agent config
        agent.budget = self.budget
        agent.cpa = self.cpa
        agent.category = self.category
        agent.reset()

        R = 0
        # consider remaining_budget issue
        for i in range(len(self.timeStepIndices)):
            # # current state
            state = (
                self.timeStepIndices[i], self.pValues[i], self.pValueSigmas[i], self.history["historyPValueInfo"],
                self.history["historyBids"], self.history["historyAuctionResult"], self.history["historyImpressionResult"],
                self.history["historyLeastWinningCost"]
            )
            # interact with env
            r_i = self.step(self.timeStepIndices[i], agent, self.history, update_action=True)
            R += r_i
            V = 0
            history = copy.deepcopy(self.history)
            agent_tmp = copy.deepcopy(agent)
            # current action
            action = (agent.w0, agent.w1)

            for j in range(i+1, len(self.timeStepIndices)):
                r_j = self.step(
                    self.timeStepIndices[j], agent_tmp, history, update_action=False)
                V += r_j
            # calculate kpi
            all_cost = agent_tmp.budget - agent_tmp.remaining_budget
            all_reward = R + V
            cpa_real = all_cost / (all_reward + 1e-10)
            cpa_constraint = agent_tmp.cpa
            all_reward_relative = all_reward/self.optimal_conv
            penalty = 20**max(0, cpa_real/cpa_constraint-1) - 1
            G = all_reward_relative - penalty

            # add experience to buffer
            agent.model.buffer.store_tuples(state, action, G)
            # update the critic and actor network
            agent.model.update()


        return








# # agent = PlayerBiddingStrategy(
# #     budget=data_loader.test_dict[key]['budget'].values[0], cpa=data_loader.test_dict[key]['CPAConstraint'].values[0], category=data_loader.test_dict[key]['advertiserCategoryIndex'].values[0])
# agent = PlayerBiddingStrategy(
#     budget=data_loader.test_dict[key]['budget'].values[0], cpa=data_loader.test_dict[key]['CPAConstraint'].values[0], category=data_loader.test_dict[key]['advertiserCategoryIndex'].values[0])

# print(agent.name)




