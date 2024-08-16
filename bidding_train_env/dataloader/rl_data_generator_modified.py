import os
import pandas as pd
import warnings
import glob

warnings.filterwarnings('ignore')


class RlDataGenerator:
    """
    RL Data Generator for RL models.
    Reads raw data and constructs training data suitable for reinforcement learning.
    """

    def __init__(self, file_folder_path="./data/traffic"):

        self.file_folder_path = file_folder_path
        self.training_data_path = self.file_folder_path + "/" + "training_data_rlData_folder"

    def batch_generate_rl_data(self):
        os.makedirs(self.training_data_path, exist_ok=True)
        csv_files = glob.glob(os.path.join(self.file_folder_path, '*.csv'))
        print(csv_files)
        training_data_list = []
        for csv_path in csv_files:
            print("开始处理文件：", csv_path)
            df = pd.read_csv(csv_path)
            df_processed = self._generate_rl_data(df)
            csv_filename = os.path.basename(csv_path)
            trainData_filename = csv_filename.replace('.csv', '-rlData.csv')
            trainData_path = os.path.join(self.training_data_path, trainData_filename)
            df_processed.to_csv(trainData_path, index=False)
            training_data_list.append(df_processed)
            del df, df_processed
            print("处理文件成功：", csv_path)
        combined_dataframe = pd.concat(training_data_list, axis=0, ignore_index=True)
        combined_dataframe_path = os.path.join(self.training_data_path, "training_data_all-rlData.csv")
        combined_dataframe.to_csv(combined_dataframe_path, index=False)
        print("整合多天训练数据成功；保存至:", combined_dataframe_path)

    def _generate_rl_data(self, df):
        """
        Construct a DataFrame in reinforcement learning format based on the raw data.

        Args:
            df (pd.DataFrame): The raw data DataFrame.

        Returns:
            pd.DataFrame: The constructed training data in reinforcement learning format.
        """

        best_result_calc_by_lp = pd.read_csv(
            '/home/dawn/NeurIPS_Auto_Bidding_General_Track_Baseline/saved_model/customLpTest/best_results_calc_by_lp.csv')


        training_data_rows = []
        for (
                deliveryPeriodIndex, advertiserNumber, advertiserCategoryIndex, budget,
                CPAConstraint), group in df.groupby(
            ['deliveryPeriodIndex', 'advertiserNumber', 'advertiserCategoryIndex', 'budget', 'CPAConstraint']):


            best_score = best_result_calc_by_lp[(best_result_calc_by_lp.deliveryPeriodIndex ==
                                                deliveryPeriodIndex) & (best_result_calc_by_lp.advertiserNumber == advertiserNumber)]['score'].values[0]
            # 
            if best_score < 1:
                best_score = 1

            group = group.sort_values('timeStepIndex')
            group['realCost'] = group['isExposed']*group['cost']

            #########   #################
            #### history impression #####
            #########   #################
            group['timeStepIndex_volume'] = group.groupby('timeStepIndex')['timeStepIndex'].transform('size')
            timeStepIndex_volume_sum = group.groupby('timeStepIndex')['timeStepIndex_volume'].first()
            historical_volume = timeStepIndex_volume_sum.cumsum().shift(1).fillna(0).astype(int)
            group['historical_volume'] = group['timeStepIndex'].map(historical_volume)
            group['future_volume'] = 50e4 - group['historical_volume']
            # last 3 volume
            last_3_timeStepIndexs_volume = timeStepIndex_volume_sum.rolling(window=3, min_periods=1).sum().shift(
                1).fillna(0).astype(int)
            group['last_3_timeStepIndexs_volume'] = group['timeStepIndex'].map(last_3_timeStepIndexs_volume)
            # last volume
            last_timeStepIndexs_volume = timeStepIndex_volume_sum.shift(1).fillna(0).astype(int)
            group['last_timeStepIndexs_volume'] = group['timeStepIndex'].map(
                last_timeStepIndexs_volume)


            #########   #################
            #### cost info #####
            #########   #################
            group['cum_realCost'] = group.groupby('timeStepIndex')['realCost'].transform('sum')
            cost_sum = group.groupby('timeStepIndex')['cum_realCost'].first()
            cum_cost = cost_sum.cumsum().fillna(0).astype(float)
            group['cum_realCost'] = group['timeStepIndex'].map(cum_cost)


            #########   #################
            #### click info #####
            #########   #################
            group['cum_click'] = group.groupby(
                'timeStepIndex')['conversionAction'].transform('sum')
            click_sum = group.groupby('timeStepIndex')['cum_click'].first()
            cum_click = click_sum.cumsum().fillna(0).astype(float)
            group['cum_click'] = group['timeStepIndex'].map(cum_click)


            #########   #################
            #### pValue info #####
            #########   #################
            group['cum_pValue'] = group.groupby(
                'timeStepIndex')['pValue'].transform('sum')
            pValue_sum = group.groupby('timeStepIndex')['cum_pValue'].first()
            cum_pValue = pValue_sum.cumsum().fillna(0).astype(float)
            group['cum_pValue'] = group['timeStepIndex'].map(cum_pValue)


            #########   #################
            #### cpa info #####
            #########   #################
            # currently, cpa is calculated based on pValue
            group['cum_cpa'] = group['cum_realCost']/group['cum_click']
            group['cpa_violation'] = group['cum_cpa']/CPAConstraint - 1
            group['cpa_violation'] = group['cpa_violation'].clip(upper=10)
            group['cum_cpa'] = group['cum_cpa'].clip(upper=100)


            #########   #################
            #### score, reward #####
            #########   #################
            # maybe need to change to real click based best score
            group['score'] = group['cum_click']*(group['cum_cpa'].apply(lambda x: min(CPAConstraint/(x+1e-6), 1)**2))/best_score
            group['score_continuous'] = group['cum_pValue'] * \
                (group['cum_cpa'].apply(lambda x: min(
                    CPAConstraint/(x+1e-6), 1)**2))/best_score

            # reward calcu
            group_agg_score = group.groupby('timeStepIndex')['score'].first()
            reward_map = group_agg_score.diff()
            group_agg_score = group_agg_score.reset_index()
            group_agg_score['reward'] = group_agg_score['timeStepIndex'].map(
                reward_map)
            group_agg_score['reward'] = group_agg_score['reward'].fillna(group_agg_score['score'])

            # reward continuous calcu
            group_agg_score_continuous = group.groupby(
                'timeStepIndex')['score_continuous'].first()
            reward_continuous_map = group_agg_score_continuous.diff()
            group_agg_score_continuous = group_agg_score_continuous.reset_index()
            group_agg_score_continuous['reward_continuous'] = group_agg_score_continuous['timeStepIndex'].map(reward_continuous_map)
            group_agg_score_continuous['reward_continuous'] = group_agg_score_continuous['reward_continuous'].fillna(
                group_agg_score_continuous['score_continuous'])


            # merge reward
            group = pd.merge(group, group_agg_score[['timeStepIndex', 'reward']], on=[
                             'timeStepIndex'], how='left')
            group = pd.merge(group, group_agg_score_continuous[['timeStepIndex', 'reward_continuous']], on=[
                             'timeStepIndex'], how='left')



            group_agg = group.groupby('timeStepIndex').agg({
                'bid': 'mean',
                'leastWinningCost': 'mean',
                'conversionAction': 'mean',
                'xi': 'mean',
                'pValue': 'mean',
                'pValueSigma': 'mean',
                'timeStepIndex_volume': 'first',
                'realCost':'mean',
                'budget':'first',
                'cpa_violation':'first'
            }).reset_index()


            for col in ['bid', 'leastWinningCost', 'conversionAction', 'xi', 'pValue', 'pValueSigma', 'realCost', 'cpa_violation']:
                group_agg[f'avg_{col}_all'] = group_agg[col].expanding().mean().shift(1)
                group_agg[f'avg_{col}_last_3'] = group_agg[col].rolling(window=3, min_periods=1).mean().shift(1)
                group_agg[f'avg_{col}_last'] = group_agg[col].shift(1)

            # diff between pred cvr and true cvr
            group_agg['avg_pValue_diff_all'] = group_agg['avg_pValue_all']-group_agg['avg_xi_all']
            group_agg['avg_pValue_diff_last_3'] = group_agg['avg_pValue_last_3'] - \
                group_agg['avg_xi_last_3']
            group_agg['avg_pValue_diff_last'] = group_agg['avg_pValue_last'] - \
                group_agg['avg_xi_last']

            # bid / leatWinningCost
            group_agg['bid_mp_ratio_all'] = group_agg['avg_bid_all'] / \
                group_agg['avg_leastWinningCost_all']
            group_agg['bid_mp_ratio_last_3'] = group_agg['avg_bid_last_3'] / \
                group_agg['avg_leastWinningCost_last_3']
            group_agg['bid_mp_ratio_last'] = group_agg['avg_bid_last'] / \
                group_agg['avg_leastWinningCost_last']

            # bid / budget
            group_agg['bid_budget_ratio_all'] = group_agg['avg_bid_all']/group_agg['budget']*48
            group_agg['bid_budget_ratio_last_3'] = group_agg['avg_bid_last_3'] / \
                group_agg['budget']*48
            group_agg['bid_budget_ratio_last'] = group_agg['avg_bid_last'] / \
                group_agg['budget']*48

            # drop unnecessary cols
            group_agg = group_agg.drop(columns=['budget'])

            # merge group_agg info
            group = group.merge(group_agg, on='timeStepIndex', suffixes=('', '_agg'))

            # 计算 realCost 和 realConversion
            realAllCost = (group['isExposed'] * group['cost']).sum()
            realAllConversion = group['conversionAction'].sum()

            for timeStepIndex in group['timeStepIndex'].unique():
                if timeStepIndex == 0:
                    bgtleft_last = 1
                


                current_timeStepIndex_data = group[group['timeStepIndex'] == timeStepIndex]

                timeStepIndexNum = 48
                current_timeStepIndex_data.fillna(0, inplace=True)
                budget = current_timeStepIndex_data['budget'].iloc[0]
                remainingBudget = current_timeStepIndex_data['remainingBudget'].iloc[0]
                timeleft = (timeStepIndexNum - timeStepIndex) / timeStepIndexNum
                bgtleft = remainingBudget / budget if budget > 0 else 0

                state_features = current_timeStepIndex_data.iloc[0].to_dict()

                state = (
                    timeleft, 
                    bgtleft,
                    bgtleft_last - bgtleft,

                    state_features['historical_volume'],
                    state_features['last_timeStepIndexs_volume'],
                    state_features['last_3_timeStepIndexs_volume'],
                    state_features['future_volume'],

                    state_features['avg_bid_all'],
                    state_features['avg_bid_last_3'],
                    state_features['avg_bid_last'],

                    state_features['avg_leastWinningCost_all'],
                    state_features['avg_leastWinningCost_last_3'],
                    state_features['avg_leastWinningCost_last'],

                    state_features['avg_conversionAction_all'],
                    state_features['avg_conversionAction_last_3'],
                    state_features['avg_conversionAction_last'],

                    state_features['avg_xi_all'],
                    state_features['avg_xi_last_3'],
                    state_features['avg_xi_last'],

                    state_features['avg_pValue_all'],
                    state_features['avg_pValue_last_3'],
                    state_features['avg_pValue_last'],

                    state_features['avg_pValueSigma_all'],
                    state_features['avg_pValueSigma_last_3'],
                    state_features['avg_pValueSigma_last'],

                    state_features['avg_cpa_violation_all'],
                    state_features['avg_cpa_violation_last_3'],
                    state_features['avg_cpa_violation_last'],

                    state_features['avg_pValue_diff_all'],
                    state_features['avg_pValue_diff_last_3'],
                    state_features['avg_pValue_diff_last'],

                    state_features['bid_mp_ratio_all'],
                    state_features['bid_mp_ratio_last_3'],
                    state_features['bid_mp_ratio_last'],

                    state_features['pValue_agg'],
                    state_features['pValueSigma_agg'],
                    state_features['timeStepIndex_volume_agg']

                )

                bgtleft_last = bgtleft

                # TODO: adjust the action definition
                total_bid = current_timeStepIndex_data['bid'].sum()
                total_value = current_timeStepIndex_data['pValue'].sum()
                action = total_bid / total_value if total_value > 0 else 0

                # reward
                reward = current_timeStepIndex_data['reward'].iloc[0]
                reward_continuous = current_timeStepIndex_data['reward_continuous'].iloc[0]

                # reward = current_timeStepIndex_data[current_timeStepIndex_data['isExposed'] == 1][
                #     'conversionAction'].sum()
                # reward_continuous = current_timeStepIndex_data[current_timeStepIndex_data['isExposed'] == 1][
                #     'pValue'].sum()

                done = 1 if timeStepIndex == timeStepIndexNum - 1 or current_timeStepIndex_data['isEnd'].iloc[
                    0] == 1 else 0

                training_data_rows.append({
                    'deliveryPeriodIndex': deliveryPeriodIndex,
                    'advertiserNumber': advertiserNumber,
                    'advertiserCategoryIndex': advertiserCategoryIndex,
                    'budget': budget,
                    'CPAConstraint': CPAConstraint,
                    'realAllCost': realAllCost,
                    'realAllConversion': realAllConversion,
                    'timeStepIndex': timeStepIndex,
                    'state': state,
                    'action': action,
                    'reward': reward,
                    'reward_continuous': reward_continuous,
                    'done': done
                })

        training_data = pd.DataFrame(training_data_rows)
        training_data = training_data.sort_values(by=['deliveryPeriodIndex', 'advertiserNumber', 'timeStepIndex'])
        

        training_data['next_state'] = training_data.groupby(['deliveryPeriodIndex', 'advertiserNumber'])['state'].shift(
            -1)
        training_data.loc[training_data['done'] == 1, 'next_state'] = None
        return training_data


def generate_rl_data():
    file_folder_path = "./data/traffic"
    data_loader = RlDataGenerator(file_folder_path=file_folder_path)
    data_loader.batch_generate_rl_data()


if __name__ == '__main__':
    generate_rl_data()
