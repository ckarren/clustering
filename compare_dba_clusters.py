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
X_train = TimeSeriesScalerMeanVariance().fit_transform(use_df)

n_cluster = 5
X1_train = ut.groupby_season(use_df)
X1_train = X1_train.T
X_train = to_time_series_dataset(X1_train)
dba_km = TimeSeriesKMeans(n_clusters=n_cluster,
                                  n_init=5,
                                  metric='dtw',
                                  verbose=True,
                                  max_iter_barycenter=10)
y_pred = dba_km.fit_predict(X_train)
df = pd.DataFrame(list(zip(list(use_df.columns), dba_km.labels_,
                      columns=['User', 'DBA cluster')
df.to_csv(f'5_clusters_monthly_results.csv')

#  df2 = pd.DataFrame(list(zip(sil_coef, inertia)),
                   #  columns=['Silhouette score', 'Inertia'])
#  df2.to_csv('silhouette_score.csv')
