from joblib import Parallel, delayed
import os
import pandas as pd
import glob
import pyomo.environ as pe
import numpy as np
import logging
import gc
from tqdm import tqdm


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(name)s] [%(filename)s(%(lineno)d)] [%(levelname)s] %(message)s"
)


def optimize(data, B, cpa, deliveryPeriodIndex, advertiserCategoryIndex, advertiserNumber):
    # it is faster to solve the primal problem directly
    data = data.reset_index(drop=True)
    data['x'] = 1

    idx = []
    # market price
    c = {}
    # decision variable
    x = {}
    # cvr
    pvalue = {}
    idx = list(data.index)
    c = data['leastWinningCost'].to_dict()
    pvalue = data['pValue'].to_dict()
    x = data['x'].to_dict()

    model = pe.ConcreteModel()
    model.dual = pe.Suffix(direction=pe.Suffix.IMPORT)
    model.idx = pe.Set(initialize=idx)
    model.B = pe.Param(initialize=B)
    model.cpa = pe.Param(initialize=cpa)
    model.c = pe.Param(model.idx, initialize=c)
    model.pvalue = pe.Param(model.idx, initialize=pvalue)
    # model.x = pe.Var(model.idx, initialize=x, bounds=[0, 1])
    model.x = pe.Var(model.idx, initialize=x, domain=pe.Binary)

    del data, idx, c, x, pvalue
    gc.collect()

    def _obj_rule(m):
        res = 0
        for i in model.idx:
            res += m.x[i]*m.pvalue[i]
        return res

    model.obj = pe.Objective(rule=_obj_rule, sense=pe.maximize)

    def budget_rule(model):
        left = 0
        for i in model.idx:
            left += model.x[i]*model.c[i]
        return left <= model.B

    def cpa_rule(model):
        left = 0
        right = 0
        for i in model.idx:
            left += model.x[i]*model.c[i]
            right += model.x[i]*model.pvalue[i]*model.cpa
        return left <= right

    model.budget_constraint = pe.Constraint(rule=budget_rule)
    model.cpa_constraint = pe.Constraint(rule=cpa_rule)

    optsolver = pe.SolverFactory('cbc')
    results = optsolver.solve(model, tee=False)
    status = 'TRUE' if results.solver.status == 'ok' and (
        results.solver.termination_condition == 'optimal' or results.solver.termination_condition == 'feasible') else 'FALSE'

    if status != 'TRUE':
        alpha = -1
        beta = -1
    else:
        alpha = np.abs(model.dual[model.budget_constraint])
        beta = np.abs(model.dual[model.cpa_constraint])
    
    return [deliveryPeriodIndex, advertiserCategoryIndex, advertiserNumber, B, cpa, status, alpha, beta, pe.value(model.obj)]


class Custom_Lp:
    """
    custom_lp model
    the bidding strategy takes form :

    """

    def __init__(self, dataPath):
        self.dataPath = dataPath

    def train(self, save_path):

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        csv_files = glob.glob(os.path.join(self.dataPath, '*.csv'))
        print(csv_files)
        csv_files = sorted(csv_files)
        # cnt = 0
        # results = []
        for i, csv_file_path in tqdm(enumerate(csv_files)):
            period_name = os.path.basename(csv_file_path).split('.')[0]
            if period_name in ['period-10', 'period-11', 'period-12', 'period-13', 'period-14']:
                continue

            df = pd.read_csv(csv_file_path)

            grouped = df.groupby(
                ['deliveryPeriodIndex', 'advertiserCategoryIndex', 'advertiserNumber',
                 'budget', 'CPAConstraint'
                 ])
            del df
            gc.collect()
            results = Parallel(n_jobs=5)(delayed(optimize)
                                          (sub_df, budget, CPAConstraint, deliveryPeriodIndex, advertiserCategoryIndex, advertiserNumber) for (deliveryPeriodIndex, advertiserCategoryIndex, advertiserNumber, budget, CPAConstraint), sub_df in grouped)

            # for (deliveryPeriodIndex, advertiserCategoryIndex, advertiserNumber, budget, CPAConstraint), sub_df in grouped:
            #     result = optimize(sub_df, budget, CPAConstraint, deliveryPeriodIndex,
            #              advertiserCategoryIndex, advertiserNumber)
            #     results.append(result)

            # results = pd.DataFrame(data=results, columns=[
            #                        'deliveryPeriodIndex', 'advertiserCategoryIndex', 'advertiserNumber', 'B', 'cpa', 'status', 'alpha', 'beta'])

            # print(f'=== {period_name} finisned saving ===')
            results = pd.DataFrame(data=results, columns=[
                'deliveryPeriodIndex', 'advertiserCategoryIndex', 'advertiserNumber', 'B', 'cpa', 'status', 'alpha', 'beta', 'obj_value'])
            results.to_csv(os.path.join(
                save_path, f'for_obj/{period_name}_bidding_param_for_obj.csv'), index=False)



