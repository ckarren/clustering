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
n_sample = 500

file_path = str('~/OneDrive - North Carolina State University/Documents/Clustering+Elasticity/InputFiles/')
use_file = file_path + 'y1_SFR_hourly.pkl'

use_df = pd.read_pickle(use_file)
use_df = ut.clean_outliers(use_df)
use_df = use_df.sample(n=n_sample, axis=1, random_state=1)
#  X_train = TimeSeriesScalerMeanVariance().fit_transform(X_train)
sil_coef = []
inertia = []
#  model = 'compare' # one of 'Kmeans', 'DBA', 'Soft-DTW', 'compare'
#  models = ['kmeans', 'DBA', 'Soft-DTW']
#  for model in models:

n_clusters = [2, 3, 4, 5]
groups = ['year', 'season', 'month']

for n_cluster in n_clusters:
    for group in groups:
        if group == 'year':
            X1_train = ut.groupby_year(use_df)#[0]
        elif group == 'season':
            X1_train = ut.groupby_season(use_df)
        elif group == 'month':
            X1_train = ut.groupby_month(use_df)
        X1_train = X1_train.T
        X_train = to_time_series_dataset(X1_train)
        for scale in [0,1]:
            if scale == 0:
                X_train = TimeSeriesScalerMeanVariance().fit_transform(X_train)
            else:
                pass
            km = TimeSeriesKMeans(n_clusters=n_cluster,
                                  metric="euclidean",
                                  verbose=True,
                                  random_state=seed)
            print(f'running k-means for {n_cluster} on {group}ly demand')
            y_pred = km.fit_predict(X_train)
            sil_coef.append(silhouette_score(X_train, y_pred))
            inertia.append(km.inertia_)
            dba_km = TimeSeriesKMeans(n_clusters=n_cluster,
                                      n_init=2,
                                      metric='dtw',
                                      verbose=True,
                                      max_iter_barycenter=10)
            print(f'running DBA k-means for {n_cluster} on {group}ly demand')
            y_pred = dba_km.fit_predict(X_train)
            sil_coef.append(silhouette_score(X_train, y_pred))
            inertia.append(dba_km.inertia_)
            sdtw_km = TimeSeriesKMeans(n_clusters=n_cluster,
                                       metric='softdtw',
                                       verbose=True,
                                       metric_params={'gamma': .01})
            print(f'running Soft-DTW k-means for {n_cluster} on {group}ly demand')
            y_pred = sdtw_km.fit_predict(X_train)
            sil_coef.append(silhouette_score(X_train, y_pred))
            inertia.append(sdtw_km.inertia_)
            df = pd.DataFrame(list(zip(list(use_df.columns), km.labels_, dba_km.labels_,
                                       sdtw_km.labels_)),
                              columns=['User', 'k-means cluster', 'DBA cluster',
                                       'SoftDTW cluster'])
            df.to_csv(f'{n_cluster}_{group}_results_{scale}.csv')

df2 = pd.DataFrame(list(zip(sil_coef, inertia)),
                   columns=['Silhouette score', 'Inertia'])
df2.to_csv('silhouette_score.csv')
