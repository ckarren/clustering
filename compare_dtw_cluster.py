import pandas as pd
import numpy as np
import utils as ut
import time
import matplotlib.pyplot as plt
from tslearn.utils import to_time_series_dataset
from tslearn.clustering import TimeSeriesKMeans, silhouette_score
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
#  import plotly.graph_objects as go
#  from plotly.subplots import make_subplots

seed = 0
np.random.seed(seed)
#  n_sample = 100
n_cluster = 5
n_init = 5
max_iter_barycenter=20
cluster_windows = [1,2,3,4,5]
file_path = str('~/OneDrive - North Carolina State University/Documents/Clustering+Elasticity/InputFiles/')
use_file = file_path + 'y1_SFR_hourly.pkl'

use_df = pd.read_pickle(use_file)
use_df = ut.clean_outliers(use_df)
#  use_df = use_df.sample(n=n_sample, axis=1, random_state=1)

X1_train = ut.groupby_year(use_df)
X1_train = X1_train.T
X_train = to_time_series_dataset(X1_train)
X_train = TimeSeriesScalerMeanVariance().fit_transform(X_train)
sz = X_train.shape[1]

for cluster_window in cluster_windows:
    begin = time.perf_counter()
    dba_km = TimeSeriesKMeans(n_clusters=n_cluster,
                              n_init=n_init,
                              metric='dtw',
                              max_iter_barycenter=max_iter_barycenter,
                              random_state=seed,
                              verbose=True,
                              metric_params={'global_constraint':'sakoe_chiba',
                                             'sakoe_chiba_radius':cluster_window},
                              n_jobs=-1)
    y_pred = dba_km.fit_predict(X_train)

    plt.figure()
    for yi in range(n_cluster):
        plt.subplot(1, n_cluster, yi + 1)
        for xx in X_train[y_pred == yi]:
            plt.plot(xx.ravel(), "k-", alpha=.2)
        plt.plot(dba_km.cluster_centers_[yi].ravel(), "r-")
        plt.xlim(0, sz)
        #  plt.ylim(0, 4)
        plt.text(0.25, .95,'Cluster %d' % (yi + 1),
                 transform=plt.gca().transAxes)
        if yi==2:
            plt.title(f'DTW on Scaled Data with Window={cluster_window}')
    #  plt.tight_layout()
#  plt.show()
#  dba_km = TimeSeriesKMeans(n_clusters=n_cluster,
                          #  n_init=5,
                          #  max_iter_barycenter=20,
                          #  metric='dtw',
                          #  random_state=seed,
                          #  verbose=True)
#  y_pred = dba_km.fit_predict(X_train)
#
#  for yi in range(n_cluster):
    #  plt.subplot(3, n_cluster, yi + n_cluster + 1)
    #  for xx in X_train[y_pred == yi]:
        #  plt.plot(xx.ravel(), "k-", alpha=.2)
    #  plt.plot(dba_km.cluster_centers_[yi].ravel(), "r-")
    #  plt.xlim(0, sz)
    #  plt.ylim(-4, 4)
    #  plt.text(0.55, 0.85,'Cluster %d' % (yi + 1),
             #  transform=plt.gca().transAxes)
    #  if yi == 1:
        #  plt.title("DBA with n_init=5")
#
#  dba_km = TimeSeriesKMeans(n_clusters=n_cluster,
                          #  n_init=60,
                          #  max_iter_barycenter=20,
                          #  metric='dtw',
                          #  verbose=True,
                          #  random_state=seed
                          #  )
#  y_pred = dba_km.fit_predict(X_train)
#
#  for yi in range(n_cluster):
    #  plt.subplot(3, n_cluster, yi + 2*n_cluster + 1)
    #  for xx in X_train[y_pred == yi]:
        #  plt.plot(xx.ravel(), "k-", alpha=.2)
    #  plt.plot(dba_km.cluster_centers_[yi].ravel(), "r-")
    #  plt.xlim(0, sz)
    #  plt.ylim(-4, 4)
    #  plt.text(0.55, 0.85,'Cluster %d' % (yi + 1),
             #  transform=plt.gca().transAxes)
    #  if yi == 1:
        #  plt.title("DBA with n_init=10")
    plt.savefig(f'{n_cluster}_clusters_DTW_scaled_r{cluster_window}.png')
    df = pd.DataFrame(list(zip(list(use_df.columns), dba_km.labels_)),
                      columns=['User', 'DBA cluster'])
    df.to_csv(f'{n_cluster}_DTW_results_scaled_r{cluster_window}.csv')

    end = time.perf_counter()
    total = (end - begin) / 60
    print(f'Clustering took {total} minutes to run')
#  df2 = pd.DataFrame(list(zip(sil_coef, inertia)),
                   #  columns=[])
#  df2.to_csv('silhouette_score.csv')