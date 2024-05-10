import pandas as pd
import numpy as np
from tslearn.utils import to_time_series_dataset
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
import plotly.graph_objects as go
from plotly.subplots import make_subplots

seed = 0 
np.random.seed(seed)
file_path = str('~/OneDrive - North Carolina State University/Documents/Clustering+Elasticity/InputFiles/')
use_file = file_path + 'MDH_SFR_Y1P1.pkl'
# load pickle file of selecte feature
X_train = pd.read_pickle(use_file)
X_train = X_train.T
print(X_train.head())
# convert to tslearn time series
X_train = to_time_series_dataset(X_train)
# pre-process data 
#  X_train = TimeSeriesScalerMeanVariance().fit_transform(X_train)

print(X_train.shape)
#  print('Euclidean k-means')
#  km = TimeSeriesKMeans(n_clusters=3,
                      #  metric="euclidean",
                      #  verbose=True,
                      #  random_state=seed)
#  y_pred = km.fit_predict(X_train)

#  fig = make_subplots(rows=3, cols=3)
#  for yi in range(3):
    #  for xx in X_train[y_pred == yi]:
        #  fig.add_trace(go.Scatter(x=np.arange(X_train.shape[1]), y=xx.ravel(),
                                 #  line_color='grey'), row=1, col=yi+1)
    #  fig.add_trace(go.Scatter(x=np.arange(X_train.shape[1]),
                             #  y=km.cluster_centers_[yi].ravel(),
                             #  line_color='darkred'), row=1,
                  #  col=yi+1)
#  print('DBA k-means')
#  dba_km = TimeSeriesKMeans(n_clusters=3,
                          #  n_init=2,
                          #  metric='dtw',
                          #  max_iter_barycenter=10)
#  y_pred = dba_km.fit_predict(X_train)

#  for yi in range(3):
    #  for xx in X_train[y_pred == yi]:
        #  fig.add_trace(go.Scatter(x=np.arange(X_train.shape[1]), y=xx.ravel(),
                                 #  line_color='grey'), row=2, col=yi+1)
    #  fig.add_trace(go.Scatter(x=np.arange(X_train.shape[1]),
                             #  y=dba_km.cluster_centers_[yi].ravel(),
                             #  line_color='darkred'),
                  #  row=2, col=yi+1)
#

#  print('Soft-DTW k-means')
#  sdtw_km = TimeSeriesKMeans(n_clusters=3,
                           #  metric='softdtw',
                           #  metric_params={'gamma': .01},
                           #  verbose=True)
#  y_pred = sdtw_km.fit_predict(X_train)

#  for yi in range(3):
    #  for xx in X_train[y_pred == yi]:
        #  fig.add_trace(go.Scatter(x=np.arange(X_train.shape[1]), y=xx.ravel(),
                                             #  line_color='grey'), row=3, col=yi+1)
    #  fig.add_trace(go.Scatter(x=np.arange(X_train.shape[1]),
                             #  y=sdtw_km.cluster_centers_[yi].ravel(),
                             #  line_color='darkred'),
                  #  row=3,col=yi+1)
#
#  fig.update_layout(title_text='Euclidean k-means')
#  fig.show()
