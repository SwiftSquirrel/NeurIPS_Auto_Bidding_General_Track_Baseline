import numpy as np
import math
import logging
from bidding_train_env.common.utils import normalize_state, normalize_reward, save_normalize_dict
from bidding_train_env.baseline.uscb.replay_buffer import ReplayBuffer
from bidding_train_env.baseline.uscb.uscb import Uscb
from bidding_train_env.environment.offline_env import OfflineEnv
import sys
import pandas as pd
import ast

np.set_printoptions(suppress=True, precision=4)
logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] [%(name)s] [%(filename)s(%(lineno)d)] [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


STATE_DIM = 16


def train_uscb_model():
    """
    Train the uscb model.
    """
    # train_data_path = "./data/traffic/training_data_rlData_folder/training_data_all-rlData.csv"
    # training_data = pd.read_csv(train_data_path)
    env = OfflineEnv()



    data_loader = TestDataLoader(file_path='./data/traffic/period-10.csv')
    env = OfflineEnv()
    keys, test_dict = data_loader.keys, data_loader.test_dict
    # why it is -1
    key = keys[-1]

    # agent = PlayerBiddingStrategy(
    #     budget=data_loader.test_dict[key]['budget'].values[0], cpa=data_loader.test_dict[key]['CPAConstraint'].values[0], category=data_loader.test_dict[key]['advertiserCategoryIndex'].values[0])
    agent = PlayerBiddingStrategy(
        budget=data_loader.test_dict[key]['budget'].values[0], cpa=data_loader.test_dict[key]['CPAConstraint'].values[0], category=data_loader.test_dict[key]['advertiserCategoryIndex'].values[0])

    print(agent.name)
    num_timeStepIndex, pValues, pValueSigmas, leastWinningCosts = data_loader.mock_data(
        key)
    cost_data = data_loader._get_cost_data_dict()[key[0]]
    cost_dict = cost_data.groupby(['timeStepIndex'])['cost'].apply(list).apply(np.array).tolist()

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
        cost = cost_dict[timeStep_index]
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

        tick_value, tick_cost, tick_status, tick_conversion, slot_status = env.simulate_ad_bidding(pValue, pValueSigma, bid,
                                                                                      leastWinningCost, cost)

        # Handling over-cost (a timestep costs more than the remaining budget of the bidding advertiser)
        over_cost_ratio = max(
            (np.sum(tick_cost) - agent.remaining_budget) / (np.sum(tick_cost) + 1e-4), 0)
        while over_cost_ratio > 0:
            pv_index = np.where(tick_status == 1)[0]
            dropped_pv_index = np.random.choice(pv_index, int(math.ceil(pv_index.shape[0] * over_cost_ratio)),
                                                replace=False)
            bid[dropped_pv_index] = 0
            tick_value, tick_cost, tick_status, tick_conversion, slot_status = env.simulate_ad_bidding(pValue, pValueSigma, bid,
                                                                                          leastWinningCost, cost)
            over_cost_ratio = max(
                (np.sum(tick_cost) - agent.remaining_budget) / (np.sum(tick_cost) + 1e-4), 0)

        win_status = (slot_status>=1).astype(int)
        agent.remaining_budget -= np.sum(tick_cost)
        rewards[timeStep_index] = np.sum(tick_conversion)
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
        # logger.info(f'Timestep Index: {timeStep_index + 1} End')
    all_reward = np.sum(rewards)
    all_cost = agent.budget - agent.remaining_budget
    cpa_real = all_cost / (all_reward + 1e-10)
    cpa_constraint = agent.cpa
    score = getScore_nips(all_reward, cpa_real, cpa_constraint)


    def safe_literal_eval(val):
        if pd.isna(val):
            return val
        try:
            return ast.literal_eval(val)
        except (ValueError, SyntaxError):
            print(ValueError)
            return val

    training_data["state"] = training_data["state"].apply(safe_literal_eval)
    training_data["next_state"] = training_data["next_state"].apply(
        safe_literal_eval)
    is_normalize = True

    if is_normalize:
        normalize_dic = normalize_state(
            training_data, STATE_DIM, normalize_indices=[13, 14, 15])
        # select use continuous reward
        training_data['reward'] = normalize_reward(
            training_data, "reward_continuous")
        # select use sparse reward
        # training_data['reward'] = normalize_reward(training_data, "reward")
        save_normalize_dict(normalize_dic, "saved_model/IQLtest")

    # Build replay buffer
    replay_buffer = ReplayBuffer()
    add_to_replay_buffer(replay_buffer, training_data, is_normalize)
    print(len(replay_buffer.memory))

    # Train model
    model = IQL(dim_obs=STATE_DIM)
    train_model_steps(model, replay_buffer)

    # Save model
    model.save_jit("saved_model/IQLtest")

    # move to device
    model.to_device()
    # Test trained model
    test_trained_model(model, replay_buffer)


def add_to_replay_buffer(replay_buffer, training_data, is_normalize):
    for row in training_data.itertuples():
        state, action, reward, next_state, done = row.state if not is_normalize else row.normalize_state, row.action, row.reward if not is_normalize else row.normalize_reward, row.next_state if not is_normalize else row.normalize_nextstate, row.done
        # ! 去掉了所有的done==1的数据
        if done != 1:
            replay_buffer.push(np.array(state), np.array([action]), np.array([reward]), np.array(next_state),
                               np.array([done]))
        else:
            replay_buffer.push(np.array(state), np.array([action]), np.array([reward]), np.zeros_like(state),
                               np.array([done]))


def train_model_steps(model, replay_buffer, step_num=20000, batch_size=100):
    for i in range(step_num):
        states, actions, rewards, next_states, terminals = replay_buffer.sample(
            batch_size)
        q_loss, v_loss, a_loss = model.step(
            states, actions, rewards, next_states, terminals)
        logger.info(
            f'Step: {i} Q_loss: {q_loss} V_loss: {v_loss} A_loss: {a_loss}')


def test_trained_model(model, replay_buffer):
    states, actions, rewards, next_states, terminals = replay_buffer.sample(
        100)
    pred_actions = model.take_actions(states)
    actions = actions.cpu().detach().numpy()
    tem = np.concatenate((actions, pred_actions), axis=1)
    print("action VS pred action:", tem)


def run_uscb():
    print(sys.path)
    """
    Run uscb model training and evaluation.
    """
    train_uscb_model()


if __name__ == '__main__':
    run_uscb()
