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
n_sample = 100

file_path = str('~/OneDrive - North Carolina State University/Documents/Clustering+Elasticity/InputFiles/')
use_file = file_path + 'y1_SFR_hourly.pkl'

use_df = pd.read_pickle(use_file)
use_df = ut.clean_outliers(use_df)
#  use_df = use_df.sample(n=n_sample, axis=1, random_state=1)
X_train = TimeSeriesScalerMeanVariance().fit_transform(use_df)
sil_coef = []
inertia = []
order = []
#  model = 'compare' # one of 'Kmeans', 'DBA', 'Soft-DTW', 'compare'
#  models = ['kmeans', 'DBA', 'Soft-DTW']
#  for model in models:

n_cluster = 3
gammas = [.01, .03, .05, 1.0, 10.0]
for gamma in gammas:
    X1_train = ut.groupby_month(use_df)
    X1_train = X1_train.T
    X_train = to_time_series_dataset(X1_train)
    #  dba_km = TimeSeriesKMeans(n_clusters=n_cluster,
                                  #  n_init=4,
                                  #  metric='dtw',
                                  #  verbose=True,
                                  #  max_iter_barycenter=max_it_bc)
        #  print(f'running DBA k-means for {n_init} inits on {group}ly demand')
    #  y_pred = dba_km.fit_predict(X_train)
    #  sil_coef.append(silhouette_score(X_train, y_pred))
    #  inertia.append(dba_km.inertia_)
    #  order.append(max_it_bc)
        #  print(silhouette_score(X_train, y_pred), dba_km.inertia_)
    sdtw_km = TimeSeriesKMeans(n_clusters=n_cluster,
                               metric='softdtw',
                               verbose=True,
                               metric_params={'gamma': gamma})
    y_pred = sdtw_km.fit_predict(X_train)
    sil_coef.append(silhouette_score(X_train, y_pred))
    inertia.append(sdtw_km.inertia_)
print(sil_coef, inertia, gamma)
    #  df = pd.DataFrame(list(zip(list(use_df.columns), dba_km.labels_,
                               #  sdtw_km.labels_)),
                      #  columns=['User', 'k-means cluster', 'DBA cluster',
                           #  'SoftDTW cluster'])
    #  df.to_csv(f'{n_cluster}_{group}_results.csv')

#  df2 = pd.DataFrame(list(zip(sil_coef, inertia)),
                   #  columns=['Silhouette score', 'Inertia'])
#  df2.to_csv('silhouette_score.csv')
