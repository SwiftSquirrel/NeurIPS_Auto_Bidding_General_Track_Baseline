import pandas as pd
import numpy as np
import os
from bidding_train_env.strategy.base_bidding_strategy import BaseBiddingStrategy

def sigmoid(x, scale=0.6, coef=1):
    return coef*scale * (1 / (1 + np.exp(-x)) - 0.5)



class CustomLpBiddingStrategy(BaseBiddingStrategy):
    """
    CustomBidding Strategy
    """

    def __init__(self, budget=750.0, name="CustomLpBiddingStrategy", cpa=8, category=1):
        super().__init__(budget, name, cpa, category)
        self.total_volumn = 499977
        self.history_volumn = 0
        self.history_spend = 0
        self.history_click = 0
        file_name = os.path.dirname(os.path.realpath(__file__))
        dir_name = os.path.dirname(file_name)
        dir_name = os.path.dirname(dir_name)
        param_dir = os.path.join(dir_name, "saved_model", "customLpTest")
        self.category = category
        self.param_data = pd.read_csv(os.path.join(param_dir, 'bidding_param.csv'))
        self.param_data['advertiserCategoryIndex'] = self.param_data['advertiserCategoryIndex'].astype(float)
        self.param_data['B'] = self.param_data['B'].astype(
            float)
        self.param_data['cpa'] = self.param_data['cpa'].astype(
            float)

        param_condi = (self.param_data.advertiserCategoryIndex == float(category)) & (
            self.param_data.B == float(budget)) & (self.param_data.cpa == float(cpa))
        self.alpha = self.param_data[param_condi]['alpha'].values[0]
        self.beta = self.param_data[param_condi]['beta'].values[0]
        self.w0 = 1/(self.alpha + self.beta)
        self.w1 = self.beta/(self.alpha + self.beta)



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

        bids = self.w0*pValues + self.w1*self.cpa*pValues
        return bids

