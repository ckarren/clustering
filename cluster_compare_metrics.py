import pandas as pd
import numpy as np
import utils as ut
from tslearn.utils import to_time_series_dataset
from tslearn.clustering import TimeSeriesKMeans, silhouette_score
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, \
    TimeSeriesResampler
from tslearn.datasets import CachedDatasets
import plotly.graph_objects as go
from plotly.subplots import make_subplots

seed = 0
np.random.seed(seed)

file_path = str('~/OneDrive - North Carolina State University/Documents/Clustering+Elasticity/InputFiles/')
use_file = file_path + 'y1_SFR_hourly.pkl'

use_df = pd.read_pickle(use_file)
use_df = ut.clean_outliers(use_df)
#  X_train = TimeSeriesScalerMeanVariance().fit_transform(X_train)
sil_coef = []
inertia = []
#  model = 'compare' # one of 'Kmeans', 'DBA', 'Soft-DTW', 'compare'
#  models = ['kmeans', 'DBA', 'Soft-DTW']
#  for model in models:
n_init = 5
max_iter_barycenter = 20
n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10]
groups = ['year', 'season', 'month']

for group in groups:
    if group == 'year':
        X1_train = ut.groupby_year(use_df)#[0]
    elif group == 'season':
        X1_train = ut.groupby_season(use_df)
    elif group == 'month':
        X1_train = ut.groupby_month(use_df)
    X1_train = X1_train.T
    X_train = to_time_series_dataset(X1_train)
    X_train = TimeSeriesScalerMeanVariance().fit_transform(X_train)
    for n_cluster in n_clusters:
        dba_km = TimeSeriesKMeans(
                    n_clusters=n_cluster,
                    n_init=n_init,
                    metric='dtw',
                    verbose=True,
                    max_iter_barycenter=max_iter_barycenter,
                    random_state=seed,
                    metric_params={
                        'global_constraint': 'sakoe_chiba',
                        'sakoe_chiba_radius': 2
                    }
        )
        y_pred = dba_km.fit_predict(X_train)
        sil_coef.append(silhouette_score(X_train, y_pred))
        inertia.append(dba_km.inertia_)
        df = pd.DataFrame(list(zip(list(use_df.columns), 
                                        dba_km.labels_),
                                columns=['User', 'DBA cluster']
                            )
                          )
        df.to_csv(f'{n_cluster}_{group}_results_{scale}.csv')

df2 = pd.DataFrame(list(zip(sil_coef, inertia)),
