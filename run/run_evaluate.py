import matplotlib.pyplot as plt
import numpy as np
import math
import logging
import bidding_train_env
from bidding_train_env.strategy import PlayerBiddingStrategy
from bidding_train_env.dataloader.test_dataloader import TestDataLoader
from bidding_train_env.environment.offline_env import OfflineEnv
import glob
import os
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(name)s] [%(filename)s(%(lineno)d)] [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# np.random.seed(1)

def getScore_nips(reward, cpa, cpa_constraint):
    beta = 2
    penalty = 1
    if cpa > cpa_constraint:
        coef = cpa_constraint / (cpa + 1e-10)
        penalty = pow(coef, beta)
    return penalty * reward


def run_test():
    """
    offline evaluation
    """

    data_loader = TestDataLoader(file_path='./data/traffic/period-10.csv')
    env = OfflineEnv()
    keys, test_dict = data_loader.keys, data_loader.test_dict
    key = keys[0]

    # agent = PlayerBiddingStrategy(
    #     budget=data_loader.test_dict[key]['budget'].values[0], cpa=data_loader.test_dict[key]['CPAConstraint'].values[0], category=data_loader.test_dict[key]['advertiserCategoryIndex'].values[0])
    agent = PlayerBiddingStrategy(
        budget=data_loader.test_dict[key]['budget'].values[0], cpa=data_loader.test_dict[key]['CPAConstraint'].values[0], category=data_loader.test_dict[key]['advertiserCategoryIndex'].values[0])

    print(agent.name)
    num_timeStepIndex, pValues, pValueSigmas, leastWinningCosts = data_loader.mock_data(
        key)
    rewards = np.zeros(num_timeStepIndex)
    history = {
        'historyBids': [],
        'historyAuctionResult': [],
        'historyImpressionResult': [],
        'historyLeastWinningCost': [],
        'historyPValueInfo': []
    }

    for timeStep_index in range(num_timeStepIndex):
        # logger.info(f'Timestep Index: {timeStep_index + 1} Begin')

        pValue = pValues[timeStep_index]
        pValueSigma = pValueSigmas[timeStep_index]
        leastWinningCost = leastWinningCosts[timeStep_index]

        if agent.remaining_budget < env.min_remaining_budget:
            bid = np.zeros(pValue.shape[0])
        else:

            bid = agent.bidding(timeStep_index, pValue, pValueSigma, history["historyPValueInfo"],
                                history["historyBids"],
                                history["historyAuctionResult"], history["historyImpressionResult"],
                                history["historyLeastWinningCost"])

        tick_value, tick_cost, tick_status, tick_conversion = env.simulate_ad_bidding(pValue, pValueSigma, bid,
                                                                                    leastWinningCost)

        # Handling over-cost (a timestep costs more than the remaining budget of the bidding advertiser)
        over_cost_ratio = max(
            (np.sum(tick_cost) - agent.remaining_budget) / (np.sum(tick_cost) + 1e-4), 0)
        while over_cost_ratio > 0:
            pv_index = np.where(tick_status == 1)[0]
            dropped_pv_index = np.random.choice(pv_index, int(math.ceil(pv_index.shape[0] * over_cost_ratio)),
                                                replace=False)
            bid[dropped_pv_index] = 0
            tick_value, tick_cost, tick_status, tick_conversion = env.simulate_ad_bidding(pValue, pValueSigma, bid,
                                                                                        leastWinningCost)
            over_cost_ratio = max(
                (np.sum(tick_cost) - agent.remaining_budget) / (np.sum(tick_cost) + 1e-4), 0)

        agent.remaining_budget -= np.sum(tick_cost)
        rewards[timeStep_index] = np.sum(tick_conversion)
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
        # logger.info(f'Timestep Index: {timeStep_index + 1} End')
    all_reward = np.sum(rewards)
    all_cost = agent.budget - agent.remaining_budget
    cpa_real = all_cost / (all_reward + 1e-10)
    cpa_constraint = agent.cpa
    score = getScore_nips(all_reward, cpa_real, cpa_constraint)

    cols=['reward','cpa','score']
    

    # plot budget and cpa based on history 
    # get spend ref at each step

    # def plot(history, budget, cpa_constraint, kp, kd, ki):
    #     total_volumn = 499977

    #     def _plot_budget(time_step, ref_budget, budget, kp, kd, ki):
    #         # 绘制ref budget
    #         plt.figure(figsize=(20, 8))
    #         plt.plot(time_step, ref_budget, marker='.', label='ref budget', color='red')
    #         # 在同一张图上绘制 budget
    #         plt.plot(time_step, budget, marker='.', 
    #                  label='budget', color='blue')
    #         # 添加图例
    #         plt.legend()
    #         # 添加标题和坐标轴标签
    #         plt.title('budget control')
    #         plt.xlabel('time step')
    #         plt.ylabel('budget')
    #         plt.xticks(time_step, time_step)

    #         # 显示图形
    #         plt.show()
    #         # plt.savefig(f'pid_plots/w2/budget_{kp}_{kd}_{ki}.png')

    #     def _plot_cpa(time_step, cpa_constraint, cpa, kp, kd, ki):
    #         # 绘制ref budget
    #         plt.figure(figsize=(20, 8))
    #         plt.plot(time_step, cpa_constraint, marker='.',
    #                  label='ref cpa', color='red')
    #         # 在同一张图上绘制 budget
    #         plt.plot(time_step, cpa, label='cpa', marker='.',  color='blue')
    #         # 添加图例
    #         plt.legend()
    #         # 添加标题和坐标轴标签
    #         plt.title('cpa control')
    #         plt.xlabel('time step')
    #         plt.ylabel('cpa')
    #         plt.xticks(time_step, time_step)

    #         # 显示图形
    #         plt.show()
    #         # plt.savefig(f'pid_plots/w2/cpa_{kp}_{kd}_{ki}.png')

    #     volumn = []
    #     click = []
    #     spend = []
    #     ref_spend = []
    #     cpa = []
    #     time_step = []

    #     for idx, (last_step_PValueInfo, last_step_AuctionResult, last_step_ImpressionResult) in enumerate(zip(history['historyPValueInfo'], history['historyAuctionResult'], history['historyImpressionResult'])):
    #         time_step.append(idx)
    #         last_step_volumn = last_step_PValueInfo.shape[0]
    #         last_step_click = last_step_ImpressionResult[:, -1].sum()
    #         last_step_spend = last_step_AuctionResult[:, -1].sum()
    #         if idx == 0:
    #             # volumn
    #             volumn.append(last_step_volumn)
    #             # click
    #             click.append(last_step_click)
    #             # spend
    #             spend.append(last_step_spend)
    #             # ref spend
    #             ref_spend.append(budget/total_volumn*volumn[-1])
    #             # cpa
    #             cpa.append(spend[-1]/click[-1])
    #         else:
    #             # volumn
    #             volumn.append(volumn[-1]+last_step_volumn)
    #             # click
    #             click.append(click[-1]+last_step_click)
    #             # spend
    #             spend.append(spend[-1]+last_step_spend)
    #             # ref spend
    #             ref_spend.append(budget/total_volumn*volumn[-1])
    #             # cpa
    #             cpa.append(spend[-1]/click[-1])
            
    #     cpa_constraint = [cpa_constraint for _ in volumn]
    #     _plot_budget(time_step, ref_spend, spend, kp, kd, ki)
    #     _plot_cpa(time_step, cpa_constraint, cpa, kp, kd, ki)
    
    # plot(history, agent.budget, cpa_constraint, kp, kd, ki)

    # # 绘制w0
    # plt.figure(figsize=(20, 8))
    # plt.plot([i for i in range(len(agent.w0_list))], agent.w0_list, marker='x',
    #          label='w0', color='red')
    # # 添加图例
    # plt.legend()
    # # 添加标题和坐标轴标签
    # plt.xlabel('time step')
    # plt.ylabel('w0')
    # plt.xticks([i for i in range(len(agent.w0_list))],
    #            [i for i in range(len(agent.w0_list))])
    # # 显示图形
    # plt.show()
    # # plt.savefig(f'pid_plots/w2/w0_{kp}_{kd}_{ki}.png')

    # # 绘制w1
    # plt.figure(figsize=(20, 8))
    # plt.plot([i for i in range(len(agent.w1_list))], agent.w1_list, marker='x',
    #          label='w1', color='red')
    # # 添加图例
    # plt.legend()
    # # 添加标题和坐标轴标签
    # plt.xlabel('time step')
    # plt.ylabel('w1')
    # plt.xticks([i for i in range(len(agent.w1_list))],
    #            [i for i in range(len(agent.w1_list))])
    # # 显示图形
    # plt.show()
    # # plt.savefig(f'pid_plots/w2/w1_{kp}_{kd}_{ki}.png')

    # # 绘制bids
    # plt.figure(figsize=(20, 8))
    # plt.plot([i for i in range(len(agent.bid_list))], agent.bid_list, marker='x',
    #          label='bids', color='red')
    # # 添加图例
    # plt.legend()
    # # 添加标题和坐标轴标签
    # plt.xlabel('time step')
    # plt.ylabel('bid')
    # plt.xticks([i for i in range(len(agent.w0_list))],
    #            [i for i in range(len(agent.w0_list))])
    # # 显示图形
    # plt.show()
    # # plt.savefig(f'pid_plots/w2/bids_{kp}_{kd}_{ki}.png')
    # plt.close()


    logger.info(f'Total Reward: {all_reward}')
    logger.info(f'Total budget: {agent.budget}')
    logger.info(f'Total Cost: {all_cost}')
    logger.info(f'CPA-real: {cpa_real}')
    logger.info(f'CPA-constraint: {cpa_constraint}')
    logger.info(f'Score: {score}')




if __name__ == '__main__':
    run_test()
