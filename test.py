from bidding_train_env.strategy import PlayerBiddingStrategy
from bidding_train_env.dataloader.test_dataloader import TestDataLoader
from bidding_train_env.environment.offline_env import OfflineEnv
import glob
import os
import pandas as pd


data = pd.read_csv('./data/traffic/period-10.csv')

print(data)
data = data[['advertiserCategoryIndex', 'budget', 'CPAConstraint']].drop_duplicates()


param_data = pd.read_csv(
    '/home/dawn/NeurIPS_Auto_Bidding_General_Track_Baseline/saved_model/customLpTest/bidding_param.csv')


for idx, row in data.iterrows():
    param_condi = (param_data.advertiserCategoryIndex == row['advertiserCategoryIndex']) & (
        param_data.B == row['budget']) & (param_data.cpa == row['CPAConstraint'])

    alpha = param_data[param_condi]['alpha'].values[0]
    beta = param_data[param_condi]['beta'].values[0]
    print('ok')

# agent = PlayerBiddingStrategy(
#     budget=data_loader.test_dict[key]['budget'].values[0], cpa=data_loader.test_dict[key]['CPAConstraint'].values[0], category=data_loader.test_dict[key]['advertiserCategoryIndex'].values[0])
