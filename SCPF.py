import pandas as pd
import scipy.sparse as ss
from sklearn.model_selection import KFold
from pathlib import Path
from SumConditionedPF import em_scpf
from Utils import *

# data_list = ['MovieLens_100K', 'MovieLens_1M']
data_list = ['BX_5']
n_factor_list = list(range(20, 101, 20))

MAX_ITER = 1000
PRINT_PERIOD = 10
CONVERGENCE_DIFF = 1e3

# num of decompositions
K = 2

a = b = 1e3
for data_name in data_list:

    df = pd.read_csv('/'.join(['D:/data', data_name + '.csv']))

    X = ss.coo_matrix((df.Count, (df.UserId - 1, df.ItemId - 1)), shape=(df.UserId.max(), df.ItemId.max())).toarray()
    # M = ss.coo_matrix((np.repeat(1, df.shape[0]), (df.UserId - 1, df.ItemId - 1)), shape=(df.UserId.max(), df.ItemId.max())).toarray()

    for R in n_factor_list:
        save_folder = '/'.join(['D:', 'result', data_name, 'SCPF', str(R) + 'factor'])

        kf = KFold(n_splits=5, shuffle=True, random_state=1234)

        iter = 1

        for train_idcs, test_idcs in kf.split(df):
            # save_folder_iter = save_folder + '/' + str(iter) + '-th_val'
            save_folder_iter = '/'.join([save_folder, str(iter) + '-th_val'])

            train = df.iloc[train_idcs]
            test = df.iloc[test_idcs]
            users_train = set(train.UserId)
            items_train = set(train.ItemId)
            test = test.loc[test.UserId.isin(users_train) & test.ItemId.isin(items_train)]

            Path(save_folder_iter).mkdir(parents=True, exist_ok=True)
            # train.to_csv(save_folder_iter + '/train.csv')
            test.to_csv(save_folder_iter + '/test.csv')

            X = ss.coo_matrix((train.Count, (train.UserId - 1, train.ItemId - 1)),
                              shape=(train.UserId.max(), train.ItemId.max())).toarray()
            M = ss.coo_matrix((np.repeat(1, train.shape[0]), (train.UserId - 1, train.ItemId - 1)),
                              shape=(train.UserId.max(), train.ItemId.max())).toarray()

            I, J = X.shape
            N_max = X.max() * np.ones((I, J))

            X_new = np.zeros((K, I, J))
            X_new[0] = X.copy()
            X_new[1] = X.max() - X_new[0]

            Mask = np.zeros((K, I, J))
            Mask[0] = M.copy()
            Mask[1] = M.copy()

            W = np.random.gamma(a, b / a, (K, I, R))
            H = np.random.gamma(a, b / a, (K, R, J))

            W, H, logP = em_scpf(N_max, X_new, Mask, W, H, A=0.001, MAX_ITER=MAX_ITER, PRINT_PERIOD=PRINT_PERIOD)
            X_hat = np.array([np.dot(W[k], H[k]) for k in range(K)])

            save_array(save_folder_iter + '/X_hat.txt', X_hat)
            save_list(save_folder_iter + '/logP.txt', logP)

            iter += 1