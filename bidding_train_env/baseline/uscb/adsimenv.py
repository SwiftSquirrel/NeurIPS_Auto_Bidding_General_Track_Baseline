import random
import pandas as pd
from bidding_train_env.dataloader.test_dataloader import TestDataLoader
from bidding_train_env.environment.offline_env import OfflineEnv
import numpy as np
import math
import itertools


class adSimEnv:

    def __init__(self, data_paths, adverNums):
        self.data_paths = data_paths
        self.adverNums = adverNums
        self.env = OfflineEnv()
        self.optimal_convs = pd.read_csv(
            '/home/dawn/NeurIPS_Auto_Bidding_General_Track_Baseline/saved_model/customLpTest/for_obj/bidding_param_for_obj.csv')
        

    def reset(self):
        """ reset env """
        random.shuffle(self.data_paths)
        random.shuffle(self.adverNums)
        self.data_path_adverNum_product = list(
            itertools.product(self.data_paths, self.adverNums)
        )


    def step(self, timeStepIndex, bid, history):
        if self.remaining_budget < self.env.min_remaining_budget:
            bid = np.zeros(self.pValues[0].shape[0])
        
        pValue = self.pValues[timeStepIndex]
        pValueSigma = self.pValueSigmas[timeStepIndex]
        leastWinningCost = self.leastWinningCosts[timeStepIndex]
        # get cost data of each position
        cost = self.cost_dict[timeStepIndex]

        # interact with env
        tick_value, tick_cost, tick_status, tick_conversion, slot_status = self.env.simulate_ad_bidding(pValue, pValueSigma, bid, leastWinningCost, cost)

        # Handling over-cost (a timestep costs more than the remaining budget of the bidding advertiser)
        over_cost_ratio = max(
            (np.sum(tick_cost) - self.remaining_budget) / (np.sum(tick_cost) + 1e-4), 0)
        while over_cost_ratio > 0:
            pv_index = np.where(tick_status == 1)[0]
            dropped_pv_index = np.random.choice(pv_index, int(math.ceil(pv_index.shape[0] * over_cost_ratio)),
                                                replace=False)
            bid[dropped_pv_index] = 0
            tick_value, tick_cost, tick_status, tick_conversion, slot_status = self.env.simulate_ad_bidding(pValue, pValueSigma, bid,
                                                                                               leastWinningCost, cost)
            over_cost_ratio = max(
                (np.sum(tick_cost) - self.remaining_budget) / (np.sum(tick_cost) + 1e-4), 0)

        win_status = (slot_status>=1).astype(int)
        self.remaining_budget -= np.sum(tick_cost)
        temHistoryPValueInfo = [(pValue[i], pValueSigma[i])
                                for i in range(pValue.shape[0])]
        history["historyPValueInfo"].append(np.array(temHistoryPValueInfo))
        history["historyBids"].append(bid)
        history["historyLeastWinningCost"].append(leastWinningCost)
        temAuctionResult = np.array(
            [(win_status[i], slot_status[i], tick_cost[i]) for i in range(tick_status.shape[0])])
        history["historyAuctionResult"].append(temAuctionResult)
        temImpressionResult = np.array(
            [(tick_status[i], tick_conversion[i]) for i in range(pValue.shape[0])])
        history["historyImpressionResult"].append(temImpressionResult)


        next_timeStepIndex = timeStepIndex + 1
        if next_timeStepIndex < self.num_timeStepIndex:
            # calculate next state
            next_state = (
                self.timeStepIndices[timeStepIndex+1], self.pValues[timeStepIndex+1], self.pValueSigmas[timeStepIndex+1], self.history["historyPValueInfo"],
                self.history["historyBids"], self.history["historyAuctionResult"], self.history["historyImpressionResult"],
                self.history["historyLeastWinningCost"]
            )
        else:
            next_state = None

        return np.sum(tick_conversion), next_state
    

    def config(self):
        # config simulation
        data_path, adverNum = self.data_path_adverNum_product.pop()
        data_loader = TestDataLoader(file_path=data_path)
        keys, test_dict = data_loader.keys, data_loader.test_dict
        key = keys[adverNum]
        period = key[0]

        self.budget = test_dict[key]['budget'].values[0]
        self.remaining_budget = self.budget
        self.category = test_dict[key]['advertiserCategoryIndex'].values[0]
        self.cpa = test_dict[key]['CPAConstraint'].values[0]

        self.optimal_convs = self.optimal_convs[self.optimal_convs.deliveryPeriodIndex == period]['obj_value'].values[0]

        num_timeStepIndex, pValues, pValueSigmas, leastWinningCosts = data_loader.mock_data(key)
        self.num_timeStepIndex = num_timeStepIndex
        self.pValues = pValues
        self.pValueSigmas = pValueSigmas
        self.leastWinningCosts = leastWinningCosts

        # cost data
        cost_data = data_loader._get_cost_data_dict()[period]
        self.cost_dict = cost_data.groupby(['timeStepIndex'])['cost'].apply(list).apply(np.array).tolist()

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
