import numpy as np
import torch
import pickle
from bidding_train_env.strategy.base_bidding_strategy import BaseBiddingStrategy
import os


class IqlBiddingStrategy(BaseBiddingStrategy):
    """
    IQL Strategy
    """

    def __init__(self, budget=100, name="Iql-PlayerStrategy", cpa=2, category=1):
        super().__init__(budget, name, cpa, category)

        file_name = os.path.dirname(os.path.realpath(__file__))
        dir_name = os.path.dirname(file_name)
        dir_name = os.path.dirname(dir_name)
        model_path = os.path.join(dir_name,"saved_model","IQLtest","iql_model.pth")
        dict_path = os.path.join(dir_name,"saved_model","IQLtest","normalize_dict.pkl")
        self.model = torch.jit.load(model_path)
        self.bgtleft_last = 1
        with open(dict_path, 'rb') as file:
            self.normalize_dict = pickle.load(file)

    def reset(self):
        self.remaining_budget = self.budget


    def bidding(self, timeStepIndex, pValues, pValueSigmas, historyPValueInfo, historyBid,
                historyAuctionResult, historyImpressionResult, historyLeastWinningCost):
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
        history_xi = [result[:, 0] for result in historyAuctionResult]

        history_realCost = []
        history_conversion = []
        history_realCost_mean = []
        for auction_result, impression_result in zip(historyAuctionResult, historyImpressionResult):
            history_realCost.append(auction_result[:,0]*impression_result[:,0])
            history_conversion.append(impression_result[:,1])
            history_realCost_mean.append(np.mean(auction_result[:,0]*impression_result[:,0]))
        
        history_pValue = [result[:,0] for result in historyPValueInfo]
        history_pValueSigma = [result[:, 1] for result in historyPValueInfo]

        time_left = (48-timeStepIndex) / 48
        bgtleft = self.remaining_budget / self.budget if self.budget > 0 else 0

        historical_volume = sum(len(bids) for bids in historyBid) if historyBid else 0
        future_volume = 50e4 - historical_volume
        last_three_timeStepIndexs_volume = sum(
            [len(historyBid[i]) for i in range(max(0, timeStepIndex - 3), timeStepIndex)]
            ) if historyBid else 0
        last_timeStepIndexs_volume = len(historyBid[-1]) if historyBid else 0

        cum_realCost = np.cumsum([np.sum(elem) for elem in history_realCost])
        cum_click = np.cumsum([np.sum(elem) for elem in history_conversion])
        cum_pValue = np.cumsum([np.sum(elem) for elem in history_pValue])
        cum_cpa = cum_realCost/cum_click
        cpa_violation = cum_cpa/self.cpa - 1
        cpa_violation = np.clip(cpa_violation, None, 10)
        cum_cpa = np.clip(cum_cpa, None, 100)

        def mean_of_last_n_elements(history, n):
            last_three_data = history[max(0, n - 3):n]
            if len(last_three_data) == 0:
                return 0
            else:
                return np.mean([np.mean(data) for data in last_three_data])
        

        historical_bid_mean = np.mean([np.mean(bid) for bid in historyBid]) if historyBid else 0
        last_three_bid_mean = mean_of_last_n_elements(historyBid, 3)
        last_bid_mean = mean_of_last_n_elements(historyBid, 1)

        historical_LeastWinningCost_mean = np.mean(
            [np.mean(price) for price in historyLeastWinningCost]) if historyLeastWinningCost else 0
        last_three_LeastWinningCost_mean = mean_of_last_n_elements(
            historyLeastWinningCost, 3)
        last_LeastWinningCost_mean = mean_of_last_n_elements(
            historyLeastWinningCost, 1)

        historical_conversion_mean = np.mean(
            [np.mean(reward) for reward in history_conversion]) if history_conversion else 0
        last_three_conversion_mean = mean_of_last_n_elements(
            history_conversion, 3)
        last_conversion_mean = mean_of_last_n_elements(
            history_conversion, 1)

        historical_xi_mean = np.mean([np.mean(xi)
                                     for xi in history_xi]) if history_xi else 0
        last_three_xi_mean = mean_of_last_n_elements(
            history_xi, 3)
        last_xi_mean = mean_of_last_n_elements(
            history_xi, 1)

        historical_pValues_mean = np.mean(
            [np.mean(value) for value in history_pValue]) if history_pValue else 0
        last_three_pValues_mean = mean_of_last_n_elements(
            history_pValue, 3)
        last_pValues_mean = mean_of_last_n_elements(
            history_pValue, 1)

        historical_pValueSigmas_mean = np.mean(
            [np.mean(value) for value in history_pValueSigma]) if history_pValueSigma else 0
        last_three_pValueSigmas_mean = mean_of_last_n_elements(
            history_pValueSigma, 3)
        last_pValueSigmas_mean = mean_of_last_n_elements(
            history_pValueSigma, 1)
        
        # last_three_realCost_mean = mean_of_last_n_elements(history_realCost, 3)
        # last_realCost_mean = mean_of_last_n_elements(history_realCost, 1)

        historical_cpa_violation = np.mean(cpa_violation)
        last_three_cpa_violation = mean_of_last_n_elements(cpa_violation, 3)
        last_cpa_violation = mean_of_last_n_elements(cpa_violation, 1)

        # diff between pred cvr and true cvr
        historical_pValue_diff = historical_pValues_mean - historical_xi_mean
        last_three_pValue_diff = last_three_pValues_mean - last_three_xi_mean
        last_pValue_diff = last_pValues_mean - last_xi_mean

        # bid / leatWinningCost
        historical_bid_mp_ratio = historical_bid_mean/(historical_LeastWinningCost_mean + 1e-8)
        last_three_bid_mp_ratio = last_three_bid_mean/(last_three_LeastWinningCost_mean + 1e-8)
        last_bid_mp_ratio = last_bid_mean/(last_LeastWinningCost_mean+1e-8)

        # # bid / budget
        # historical_bid_budget_ratio = historical_bid_mean/self.budget*48
        # last_three_bid_budget_ratio = last_three_bid_mean/self.budget*48
        # last_bid_budget_ratio = last_bid_mean/self.budget*48

        current_pValues_mean = np.mean(pValues)
        current_pValueSigmas_mean = np.mean(pValueSigmas)
        current_pv_num = len(pValues)


        test_state = np.array([
            time_left, 
            bgtleft,
            self.bgtleft_last - bgtleft, 

            historical_volume,
            last_timeStepIndexs_volume,
            last_three_timeStepIndexs_volume,
            future_volume,

            historical_bid_mean,
            last_three_bid_mean,
            last_bid_mean,

            historical_LeastWinningCost_mean,
            last_three_LeastWinningCost_mean,
            last_LeastWinningCost_mean,

            historical_conversion_mean,
            last_three_conversion_mean,
            last_conversion_mean,

            historical_xi_mean,
            last_three_xi_mean,
            last_xi_mean,

            historical_pValues_mean,
            last_three_pValues_mean,
            last_pValues_mean,

            historical_pValueSigmas_mean,
            last_three_pValueSigmas_mean,
            last_pValueSigmas_mean,

            historical_cpa_violation,
            last_three_cpa_violation,
            last_cpa_violation,

            historical_pValue_diff,
            last_three_pValue_diff,
            last_pValue_diff,

            historical_bid_mp_ratio,
            last_three_bid_mp_ratio,
            last_bid_mp_ratio,

            current_pValues_mean,
            current_pValueSigmas_mean,
            current_pv_num

        ])
        test_state = np.nan_to_num(test_state)

        def normalize(value, min_value, max_value):
            return (value - min_value) / (max_value - min_value) if max_value > min_value else 0

        for key, value in self.normalize_dict.items():
            test_state[key] = normalize(test_state[key], value["min"], value["max"])


        test_state = torch.tensor(test_state, dtype=torch.float)
        alpha = self.model(test_state)
        alpha = alpha.cpu().numpy()
        bids = alpha * pValues

        self.bgtleft_last = bgtleft

        return bids
