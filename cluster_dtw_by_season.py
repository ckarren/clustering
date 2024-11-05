import pandas as pd
import numpy as np
import utils as ut
import time
import matplotlib.pyplot as plt
from tslearn.utils import to_time_series_dataset
from tslearn.clustering import TimeSeriesKMeans, silhouette_score
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

seed = 0
np.random.seed(seed)
n_init = 5
max_iter_barycenter=20
cluster_window = 1
n_clusters = [6, 7, 8, 9]
seasons = ['summer', 'winter']

file_path = str('../InputFiles/')
use_file1 = file_path + 'y1_SFR_hourly.pkl'
use_file2 = file_path + 'y2_SFR_hourly.pkl'

use_df1 = pd.read_pickle(use_file1)
use_df2 = pd.read_pickle(use_file2)
df_use = pd.concat([use_df1, use_df2], join='inner')
df_use = ut.clean_outliers(df_use)

X1_train = ut.groupby_season(df_use)

for cluster in n_clusters:
    for season in seasons:
        if season == 'summer':
            X1_train = X1_train.iloc[0:24,:]
        elif season == 'autumn':
            X1_train = X1_train.iloc[24:48,:]
        elif season == 'winter':
            X1_train = X1_train.iloc[48:72,:]
        elif season == 'spring':
            X1_train = X1_train.iloc[72:97,:]
        
        X1_train = X1_train.T
        X_train = to_time_series_dataset(X1_train)
        X_train = TimeSeriesScalerMeanVariance().fit_transform(X_train)
        sz = X_train.shape[1]
        dba_km = TimeSeriesKMeans(n_clusters=cluster,
                                      n_init=n_init,
                                      metric='dtw',
                                      max_iter_barycenter=max_iter_barycenter,
                                      random_state=seed,
                                      verbose=True,
                                      metric_params={'global_constraint':'sakoe_chiba',
                                                     'sakoe_chiba_radius':cluster_window},
                                      n_jobs=-1)
        y_pred = dba_km.fit_predict(X_train)

    df = pd.DataFrame(list(zip(list(df_use.columns), dba_km.labels_)),
                      columns=['User', 'DBA cluster'])
    df.to_csv(f'{cluster}_clusters_DTW_results_scaled_{season}.csv')

