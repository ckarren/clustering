import pandas as pd
import numpy as np
import utils as ut
import time
import matplotlib.pyplot as plt
from tslearn.utils import to_time_series_dataset
from tslearn.clustering import TimeSeriesKMeans, silhouette_score
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

#  seed = #  0
#  np.random.seed(seed)
#  n_sample = 1000
#  #  n_init = 2
#  #  max_iter_barycenter=20
#  cluster_window = 1
#  #  n_clusters = [6, 7, 8, 9]
#  n_clusters = 5
#  file_path = str('../InputFiles/')
#  use_file = file_path + 'y1_SFR_hourly.pkl'
#
#  use_df = pd.read_pickle(use_file)
#  use_df = ut.clean_outliers(use_df)
#  #  use_df = use_df.sample(n=n_sample, axis=1, random_state=1)
seasons = ['summer', 'autumn', 'winter', 'spring']
#  #  season = 'summer'
#  #  X1_train = ut.groupby_year(use_df)
#  begin = time.perf_counter()
#  X1_train = ut.groupby_season(use_df)
#  #  X1_train = X1_train.iloc[0:24,:]
clusters = []
for season in seasons:
    data = pd.read_csv(f'../5_clusters_output/5_clusters_DTW_results_scaled_{season}.csv', 
                        usecols=[1,2], 
                        header=0, 
                        index_col=0)
    df = pd.DataFrame(data)
    df.rename(columns={'DBA cluster': f'{season} Cluster'}, inplace=True)
    clusters.append(df)

df_all = pd.concat(clusters, axis=1, join='inner')
df_all.to_csv('5_clusters_by_season.csv')
#      X1_train = ut.groupby_season(use_df)
    #      if season == 'summer':
    #          X1_train = X1_train.iloc[0:24,:]
    #      elif season == 'autumn':
    #          X1_train = X1_train.iloc[24:48,:]
    #      elif season == 'winter':
    #          X1_train = X1_train.iloc[48:72,:]
    #      elif season == 'spring':
    #          X1_train = X1_train.iloc[72:97,:]
#
#      X1_train = X1_train.T
#      X_train = to_time_series_dataset(X1_train)
#      X_train = TimeSeriesScalerMeanVariance().fit_transform(X_train)
#      sz = X_train.shape[1]
#      dba_km = TimeSeriesKMeans(n_clusters=n_clusters,
#                                metric='dtw',
#                                random_state=seed,
#                                verbose=True,
#                                metric_params={'global_constraint':'sakoe_chiba',
#                                               'sakoe_chiba_radius':cluster_window},
#                                n_jobs=-1)
#      y_pred = dba_km.fit_predict(X_train)
#
#          #  plt.figure()
#          #  for yi in range(n_cluster):
#              #  plt.subplot(1, n_cluster, yi + 1)
#              #  for xx in X_train[y_pred == yi]:
#                  #  plt.plot(xx.ravel(), "k-", alpha=.2)
#              #  plt.plot(dba_km.cluster_centers_[yi].ravel(), "r-")
#              #  plt.xlim(0, sz)
#              #  plt.ylim(0, 4)
#              #  plt.text(0.25, .95,'Cluster %d' % (yi + 1),
#                       #  transform=plt.gca().transAxes)
#              #  if yi==2:
#                  #  plt.title(f'DTW on Scaled Yearly Data with n_clusters={n_cluster}')
#          #  plt.savefig(f'{n_cluster}_clusters_DTW_scaled_r1.png')
#      df = pd.DataFrame(list(zip(list(use_df.columns), dba_km.labels_)),
#                        columns=['User', 'DBA cluster'])
#      df.to_csv(f'5_clusters_DTW_results_scaled_{season}.csv')
#
#  #  sil_coef.append(silhouette_score(X_train, y_pred))
#  #  inertia.append(dba_km.inertia_)
#  #  df2 = pd.DataFrame(list(zip(sil_coef, inertia)),
#                 #  columns=['Silhouette coef', 'Intertia'])
#  #  df2.to_csv(f'5_clusters_silhouette_score_{season}.csv')
#
#  end = time.perf_counter()
#  total = (end - begin) / 60
#  print(f'Clustering took {total} minutes to run')
#
