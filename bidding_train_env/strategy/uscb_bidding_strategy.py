import numpy as np
import torch
import pickle
from bidding_train_env.strategy.base_bidding_strategy import BaseBiddingStrategy
import os
from bidding_train_env.baseline.uscb.uscb import Uscb


class UscbBiddingStrategy(BaseBiddingStrategy):
    """
    Uscb Strategy
    """

    def __init__(self, budget=100, name="uscb-PlayerStrategy", cpa=2, category=1, test=True):
        super().__init__(budget, name, cpa, category)

        file_name = os.path.dirname(os.path.realpath(__file__))
        dir_name = os.path.dirname(file_name)
        dir_name = os.path.dirname(dir_name)
        if test:
            model_path = os.path.join(
                dir_name, "saved_model", "uscbtest", "uscb_model.pth")
            dict_path = os.path.join(
                dir_name, "saved_model", "uscbtest", "normalize_dict.pkl")
            self.model = torch.jit.load(model_path)
        else:
            self.model = Uscb()
        # with open(dict_path, 'rb') as file:
        #     self.normalize_dict = pickle.load(file)
        self.w0 = 10
        self.w1 = 0.5
        self.w0_lb = 0.5
        self.w0_ub = 17
        self.w1_lb = 0
        self.w1_ub = 1


    def reset(self):
        self.remaining_budget = self.budget

    def bidding(self, timeStepIndex, pValues, pValueSigmas, historyPValueInfo, historyBid,
                historyAuctionResult, historyImpressionResult, historyLeastWinningCost, update_action=True):
        """
        Bids for all the opportunities in a delivery period

        parameters:
         @timeStepIndex: the index of the current decision time step.
         @pValues: the conversion action probability.
         @pValueSigmas: the prediction probability uncertainty.
         @historyPValueInfo: the history predicted value and uncertainty for each opportunity.
         @historyBid: the advertiser's history bids for each opportunity.
         @historyAuctionResult: the history auction results for each opportunity.
         @historyImpressionResult: the history impression result for each opportunity.
         @historyLeastWinningCosts: the history least wining costs for each opportunity.

        return:
            Return the bids for all the opportunities in the delivery period.
        """
        time_left = (48 - timeStepIndex) / 48
        budget_left = self.remaining_budget / self.budget if self.budget > 0 else 0
        history_xi = [result[:, 0] for result in historyAuctionResult]
        history_pValue = [result[:, 0] for result in historyPValueInfo]
        history_conversion = [result[:, 1]
                              for result in historyImpressionResult]
        historical_xi_mean = np.mean([np.mean(xi)
                                     for xi in history_xi]) if history_xi else 0
        historical_conversion_mean = np.mean(
            [np.mean(reward) for reward in history_conversion]) if history_conversion else 0
        historical_LeastWinningCost_mean = np.mean(
            [np.mean(price) for price in historyLeastWinningCost]) if historyLeastWinningCost else 0
        historical_pValues_mean = np.mean(
            [np.mean(value) for value in history_pValue]) if history_pValue else 0
        historical_bid_mean = np.mean(
            [np.mean(bid) for bid in historyBid]) if historyBid else 0

        def mean_of_last_n_elements(history, n):
            last_three_data = history[max(0, n - 3):n]
            if len(last_three_data) == 0:
                return 0
            else:
                return np.mean([np.mean(data) for data in last_three_data])

        last_three_xi_mean = mean_of_last_n_elements(history_xi, 3)
        last_three_conversion_mean = mean_of_last_n_elements(
            history_conversion, 3)
        last_three_LeastWinningCost_mean = mean_of_last_n_elements(
            historyLeastWinningCost, 3)
        last_three_pValues_mean = mean_of_last_n_elements(history_pValue, 3)
        last_three_bid_mean = mean_of_last_n_elements(historyBid, 3)


        current_pValues_mean = np.mean(pValues)
        current_pv_num = len(pValues)/50e4*48
        historical_pv_num_total = sum(len(bids)
                                    for bids in historyBid)/50e4*48 if historyBid else 0
        last_three_ticks = slice(max(0, timeStepIndex - 3), timeStepIndex)
        last_three_pv_num_total = sum(
            [len(historyBid[i]) for i in range(max(0, timeStepIndex - 3), timeStepIndex)])/50e4*48 if historyBid else 0
        test_state = np.array([
            time_left, budget_left, historical_bid_mean, last_three_bid_mean,
            historical_LeastWinningCost_mean, historical_pValues_mean, historical_conversion_mean,
            historical_xi_mean, last_three_LeastWinningCost_mean, last_three_pValues_mean,
            last_three_conversion_mean, last_three_xi_mean,
            current_pValues_mean, current_pv_num, last_three_pv_num_total,
            historical_pv_num_total
        ])

        if update_action:
            test_state = torch.tensor(test_state, dtype=torch.float)
            self.w0, self.w1 = self.model.policy(test_state)
            self.w0 = np.clip(self.w0, self.w0_lb, self.w0_ub)
            self.w1 = np.clip(self.w1, self.w1_lb, self.w1_ub)

        bids = self.w0 * pValues + self.w1*self.cpa*pValues

        return bids
