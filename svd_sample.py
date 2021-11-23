import pandas as pd

df = pd.read_csv('MovieLens_100K.csv')
df.columns = ['userID', 'itemID', 'rating']

from surprise import SVD
from surprise import SVDpp
from surprise import NMF
from surprise import KNNBaseline

n_factors = 20

# algo = SVD(n_factors = n_factors, verbose = True)
# algo = SVDpp(n_factors = n_factors, verbose = True)
# algo = NMF(n_factors = n_factors, verbose = True)
algo = KNNBaseline(n_factors = n_factors, verbose = True)

from surprise import Reader
reader = Reader(rating_scale=(1, df.rating.max()))

from surprise import Dataset
data = Dataset.load_from_df(df[['userID', 'itemID', 'rating']], reader)

from surprise.model_selection import cross_validate
cross_validate(algo, data, measures=['RMSE', 'MAE', ''], cv=5, verbose=True)

