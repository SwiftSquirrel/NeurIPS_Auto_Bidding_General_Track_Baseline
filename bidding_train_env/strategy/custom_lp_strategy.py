import pandas as pd
import numpy as np
import os
import xgboost as xgb
from bidding_train_env.strategy.base_bidding_strategy import BaseBiddingStrategy
from bidding_train_env.common.utils import PID

def sigmoid(x, scale=0.6, coef=1):
    return coef*scale * (1 / (1 + np.exp(-x)) - 0.5)

class CustomLpBiddingStrategy(BaseBiddingStrategy):
    """
    CustomBidding Strategy
    """

    def __init__(self, budget=750.8, name="CustomLpBiddingStrategy", cpa=8, category=1):
        super().__init__(budget, name, cpa, category)
        self.total_volumn = 499977
        self.history_volumn = 0
        self.history_spend = 0
        self.history_click = 0
        file_name = os.path.dirname(os.path.realpath(__file__))
        dir_name = os.path.dirname(file_name)
        dir_name = os.path.dirname(dir_name)
        model_dir = os.path.join(dir_name, "saved_model", "customLpTest")
        self.category = category

        self.model_alpha = xgb.XGBRegressor()
        self.model_alpha.load_model(
            os.path.join(model_dir, 'model_alpha.json'))
        self.model_beta = xgb.XGBRegressor()
        self.model_beta.load_model(
            os.path.join(model_dir, 'model_beta.json'))
        
        self.alpha = self.model_alpha.predict(
            pd.DataFrame(data=[[self.category, self.budget, self.cpa]], columns=[
                'advertiserCategoryIndex', 'B', 'cpa'])
        )[0]
        self.beta = self.model_beta.predict(
            pd.DataFrame(data=[[self.category, self.budget, self.cpa]], columns=[
                'advertiserCategoryIndex', 'B', 'cpa'])
        )[0]
        self.w0 = 1/(self.alpha+self.beta)
        self.w1 = self.beta/(self.alpha+self.beta)


        # self.pid_w0 = PID(Kp=0.005, Kd=0.5, Ki=0.05, name='budget')
        self.pid_w0 = PID(Kp=0.005, Kd=0.5, Ki=0.05, name='budget')
        # 0.5, 0.001, 0.005
        self.pid_w1 = PID(Kp=0.1, Kd=0.01, Ki=0.05, name='cpa')
        # self.pid_w1 = PID(Kp=w1_kp, Kd=w1_kd, Ki=w1_ki, name='cpa')

        # self.pid_w1 = PID(Kp=0, Kd=0, Ki=0, name='cpa')

        self.w0_list = [self.w0]
        self.w1_list = [self.w1]
        self.bid_list = []



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
        if timeStepIndex >= 1:
            if timeStepIndex == 18:
                print('stoped here')
            last_step_volumn = historyPValueInfo[-1].shape[0]
            self.history_volumn += last_step_volumn
            last_step_spend = historyAuctionResult[-1][:, -1].sum()
            self.history_spend += last_step_spend
            ref_spend = self.budget/self.total_volumn*self.history_volumn

            last_step_click = historyImpressionResult[-1][:, -1].sum()
            self.history_click += last_step_click

            if self.history_click==0:
                cur_cpa = 0
            else:
                cur_cpa = self.history_spend/self.history_click

            w0_update = self.pid_w0.update(
                ref_spend, self.history_spend, last_step_volumn*48/self.total_volumn)
            w0_update = sigmoid(w0_update)

            w1_update = self.pid_w1.update(
                self.cpa, cur_cpa, last_step_volumn*48/self.total_volumn)
            w1_update = sigmoid(w1_update, scale=1, coef=15/self.cpa)

            self.w0 = self.w0*(1+w0_update)
            self.w1 = self.w1*(1+w1_update)

            self.w0 = np.clip(self.w0, 0.59, 15.08)
            self.w1 = np.clip(self.w1, 0, 1)

            self.w0_list.append(self.w0)
            self.w1_list.append(self.w1)


        # bids = 1/(self.alpha + self.beta)*pValues + self.beta/(self.alpha + self.beta)*self.cpa*pValues
        bids = self.w0*pValues + self.w1*self.cpa*pValues
        self.bid_list.append(np.mean(bids))

        return bids
