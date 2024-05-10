import pandas as pd
import numpy as np
from utils import groupby_year
from tslearn.utils import to_time_series_dataset
from tslearn.clustering import TimeSeriesKMeans, silhouette_score
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, \
    TimeSeriesResampler
from tslearn.datasets import CachedDatasets
import plotly.graph_objects as go
from plotly.subplots import make_subplots

seed = 0 
np.random.seed(seed)
# X_train, y_train, X_test, y_test= UCR_UEA_datasets().load_dataset("TwoPatterns")
# print(y_train)
#  X_train, y_train, X_test, y_test = CachedDatasets().load_dataset("Trace")
#  X_train = X_train[y_train < 4]  # Keep first 3 classes
#  np.random.shuffle(X_train)
#  # Keep only 50 time series
#  X_train = TimeSeriesScalerMeanVariance().fit_transform(X_train[:50])
#  # Make time series shorter
#  X_train = TimeSeriesResampler(sz=40).fit_transform(X_train)
#  sz = X_train.shape[1]
file_path = str('~/OneDrive - North Carolina State University/Documents/Clustering+Elasticity/InputFiles/')
use_file = file_path + 'hourly_use_SFR_y1.pkl'
use_df = pd.read_pickle(use_file)
use_df = use_df.sample(n=1000, axis=1, random_state=1)
X_train = groupby_year(use_df)#[0]
X_train = X_train.T
X_train = to_time_series_dataset(X_train)
#  X_train = TimeSeriesScalerMeanVariance().fit_transform(X_train)
cluster_n = [2]
silhouette_coef = []
inertia = []
for i in cluster_n:
    km = TimeSeriesKMeans(n_clusters=i,
                          metric="euclidean",
                          verbose=True,
                          random_state=seed)
    y_pred = km.fit_predict(X_train)
    inertia.append(km.inertia_)
    silhouette_coef.append(silhouette_score(X_train,y_pred))


print(silhouette_coef, inertia)
    #  fig = make_subplots(rows=1, cols=i)
    #  for yi in range(i):
        #  for xx in X_train[y_pred == yi]:
            #  fig.add_trace(go.Scatter(x=np.arange(X_train.shape[1]), y=xx.ravel(),
                                     #  line_color='grey'),
                          #  row=1, col=yi+1)
        #  fig.add_trace(go.Scatter(x=np.arange(X_train.shape[1]),
                                 #  y=km.cluster_centers_[yi].ravel(),
                                 #  line_color='darkred'),
                      #  row=1, col=yi+1)
    #  fig.show()

#  print('DBA k-means')
#  dba_km = TimeSeriesKMeans(n_clusters=3,
#                            n_init=2,
#                            metric='dtw',
#                            max_iter_barycenter=10)
#  y_pred = dba_km.fit_predict(X_train)
#
#  for yi in range(3):
#
#      plt.subplot(3,3,yi+4)
#      for xx in X_train[y_pred == yi]:
#          plt.plot(dba_km.cluster_centers_[yi].ravel(), 'r-')
#      if yi == 1:
#          plt.title('DBA $k$-means')
#
#  print('Soft-DTW k-means')
#  sdtw_km = TimeSeriesKMeans(n_clusters=3,
#                             metric='softdtw',
#                             metric_params={'gamma': .01},
#                             verbose=True)
#  y_pred = sdtw_km.fit_predict(X_train)
#
#  for yi in range(3):
#      plt.subplots(3,3,7+yi)
#      for xx in X_train[y_pred == yi]:
#          plt.plot(sdtw_km.cluster_centers_[yi].ravel(), 'r-')
#      if yi == 1:
#          plt.title('Soft-DTW $k$-means')
#  fig.update_layout(title_text='DTW k-means')
#  fig.show()
