from ProbabilisticMatrixFactorization import PMF
import pandas as pd
from sklearn.model_selection import KFold
from pathlib import Path
from Utils import *
import time

# data_list = ['MovieLens_100K', 'MovieLens_1M']
data_list = ['jester_20']
n_factor_list = list(range(20, 101, 20))
# n_factor_list = list(range(20, 21, 20))
# save = False
save = True
max_epoch = 1000
CONVERGENCE_DIFF = 1e-3
for data_name in data_list:
    # data_name = 'MovieLens_100K'

    df = pd.read_csv('/'.join(['D:/data', data_name + '.csv']))

    for n_factor in n_factor_list:
        # n_factor = 20
        save_folder = '/'.join(['D:', 'result', data_name, 'PMF', str(n_factor) + 'factor'])
    
        kf = KFold(n_splits=5, shuffle=True, random_state=1234)
    
        iter = 1
        for train_idcs, test_idcs in kf.split(df):
            pmf = PMF()
            save_folder_iter = save_folder
            # save_folder_iter = save_folder + '/' + str(iter) + '-th_val'
    
            train = df.iloc[train_idcs]
            test = df.iloc[test_idcs]
            users_train = set(train.UserId)
            items_train = set(train.ItemId)
            test = test.loc[test.UserId.isin(users_train) & test.ItemId.isin(items_train)]
    
            test = test.values
            train = train.values

            pmf.set_params({"num_feat": n_factor, "epsilon": 1, "_lambda": 0.1, "momentum": 0.8, "maxepoch": max_epoch,
                            "num_batches": round(len(train) ** (1 / 2)),
                            "batch_size": round(len(train) ** (1 / 2)), 'CONVERGENCE_DIFF': CONVERGENCE_DIFF})

            Path(save_folder_iter).mkdir(parents=True, exist_ok=True)

            result = pmf.fit(train, test)

            if save:
                save_list(save_folder_iter + '/rmse_' + str(iter) + '-th_iter.txt', result)
    
            iter += 1