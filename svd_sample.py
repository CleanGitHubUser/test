import pandas as pd
from surprise import Reader
from surprise import Dataset
from pathlib import Path
from surprise import SVD
from surprise import NMF
from surprise import KNNBaseline
from surprise.model_selection import cross_validate
from Utils import *

# data_list = ['MovieLens_100K', 'MovieLens_1M']
data_list = ['jester_20']
n_factor_list = list(range(20, 101, 20))
for data_name in data_list:
    # data_name = 'MovieLens_1M'
    # data_name = 'MovieLens_100K'

    df = pd.read_csv('/'.join(['D:/data', data_name + '.csv']))
    df.columns = ['userID', 'itemID', 'rating']

    reader = Reader(rating_scale=(1, df.rating.max()))

    data = Dataset.load_from_df(df[['userID', 'itemID', 'rating']], reader)

    for n_factors in n_factor_list:
        # n_factors = 20
        save_folder = '/'.join(['D:', 'result', data_name, 'surprise', str(n_factors) + 'factor'])

        Path(save_folder).mkdir(parents=True, exist_ok=True)

        algo = SVD(n_factors = n_factors, verbose = True)

        result = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

        save_list('/'.join([save_folder, 'rmse_SVD.txt']), result['test_rmse'])

        algo = NMF(n_factors=n_factors, verbose=True)

        result = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

        save_list('/'.join([save_folder, 'rmse_NMF.txt']), result['test_rmse'])

        algo = KNNBaseline(n_factors=n_factors, verbose=True)

        result = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

        save_list('/'.join([save_folder, 'rmse_KNN.txt']), result['test_rmse'])