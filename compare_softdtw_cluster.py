import pandas as pd
import numpy as np
import utils as ut
import matplotlib.pyplot as plt
from tslearn.utils import to_time_series_dataset
from tslearn.clustering import TimeSeriesKMeans, silhouette_score
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
#  import plotly.graph_objects as go
#  from plotly.subplots import make_subplots

seed = 0
np.random.seed(seed)
n_cluster = 5

file_path = str('../InputFiles/')
use_file = file_path + 'y1_SFR_hourly.pkl'

use_df = pd.read_pickle(use_file)
use_df = ut.clean_outliers(use_df)
#  use_df = use_df.sample(n=n_sample, axis=1, random_state=1)
sil_coef = []
inertia = []
#  model = 'compare' # one of 'Kmeans', 'DBA', 'Soft-DTW', 'compare'
#  models = ['kmeans', 'DBA', 'Soft-DTW']
#  for model in models:

X1_train = ut.groupby_year(use_df)
X1_train = X1_train.T
X_train = to_time_series_dataset(X1_train)
X_train = TimeSeriesScalerMeanVariance().fit_transform(X_train)
sz = X_train.shape[1]
euclidean_km = TimeSeriesKMeans(n_clusters=n_cluster,
                          random_state=seed,
                          metric='euclidean',
                          verbose=True)#,
                          #  metric_params =
                          #  {'global_constraint':'sakoe_chiba',
                           #  'sakoe_chiba_radius':3})

y_pred = euclidean_km.fit_predict(X_train)

plt.figure()
for yi in range(n_cluster):
    plt.subplot(2, n_cluster, yi + 1)
    for xx in X_train[y_pred == yi]:
        plt.plot(xx.ravel(), "k-", alpha=.2)
    plt.plot(euclidean_km.cluster_centers_[yi].ravel(), "r-")
    plt.xlim(0, sz)
    #  plt.ylim(-4, 4)
    plt.text(0.55, 0.85,'Cluster %d' % (yi + 1),
             transform=plt.gca().transAxes)
    if yi == 2:
        plt.title("Euclidean")
#
dba_km = TimeSeriesKMeans(n_clusters=n_cluster,
                          random_state=seed,
                          metric='dtw',
                          verbose=True,
                          metric_params =
                          {'global_constraint':'sakoe_chiba',
                           'sakoe_chiba_radius':1}
                         )

y_pred = dba_km.fit_predict(X_train)

for yi in range(n_cluster):
    plt.subplot(2, n_cluster, yi + n_cluster + 1)
    for xx in X_train[y_pred == yi]:
        plt.plot(xx.ravel(), "k-", alpha=.2)
    plt.plot(dba_km.cluster_centers_[yi].ravel(), "r-")
    plt.xlim(0, sz)
    plt.ylim(-4, 4)
    plt.text(0.55, 0.85,'Cluster %d' % (yi + 1),
             transform=plt.gca().transAxes)
    if yi == 2:
        plt.title("DBA with window of 1")
#
#  dba_km = TimeSeriesKMeans(n_clusters=n_cluster,
                          #  random_state=seed,
                          #  metric='euclidean',
                          #  verbose=True,
                          #  metric_params={'global_constraint': 'sakoe_chiba',
                                         #  'sakoe_chiba_radius': 10}
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
        #  plt.title("DBA with window of 10")
plt.savefig('compare_eucliean_dtw1.png')
df = pd.DataFrame(list(zip(list(use_df.columns), euclidean_km.labels_, dba_km.labels_,)),
                              columns=['User', 'k-means cluster', 'DBA cluster'])
df.to_csv(f'{n_cluster}_results_euclidean_dtw1.csv')
#
#  df2 = pd.DataFrame(list(zip(sil_coef, inertia)),
                   #  columns=['Silhouette score', 'Inertia'])
#  df2.to_csv('silhouette_score.csv')
