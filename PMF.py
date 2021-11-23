from ProbabilisticMatrixFactorization import PMF

save_folder = 'D:/PMF'
import pandas as pd
df = pd.read_csv('MovieLens_100K.csv')

n_factor = 20

pmf = PMF()
pmf.set_params({"num_feat": n_factor, "epsilon": 1, "_lambda": 0.1, "momentum": 0.8, "maxepoch": 100, "num_batches": 100,
                "batch_size": 1000, 'CONVERGENCE_DIFF': 1e-3})

from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True, random_state=1234)

from pathlib import Path
from Utils import *

iter = 1
for train_idcs, test_idcs in kf.split(df):
    save_folder_iter = save_folder + '/' + str(iter) + '-th_val'

    train = df.iloc[train_idcs]
    test = df.iloc[test_idcs]
    users_train = set(train.UserId)
    items_train = set(train.ItemId)
    test = test.loc[test.UserId.isin(users_train) & test.ItemId.isin(items_train)]

    test = test.values
    train = train.values

    Path(save_folder_iter).mkdir(parents=True, exist_ok=True)

    save_list(save_folder_iter + '/rmse.txt', pmf.fit(train, test))

    iter += 1