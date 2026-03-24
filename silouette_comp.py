import pandas as pd
import numpy as np
from utils import groupby_year
from tslearn.clustering import silhouette_score

import config
from clustering_engine import ClusteringConfig, TimeSeriesClusterer
from data import DataLoader

seed = config.DEFAULT_SEED
np.random.seed(seed)


class SilhouetteComparisonRunner:
    def __init__(self, sample_size=1000):
        self.sample_size = sample_size
        self.loader = DataLoader()

    def run(self):
        use_df = self.loader.load_water_use('hourly_use_SFR_y1.pkl', clean=False)
        use_df = use_df.sample(n=self.sample_size, axis=1, random_state=1)
        train_frame = groupby_year(use_df).T
        cluster_n = [2]
        silhouette_coef = []
        inertia = []

        for n_clusters in cluster_n:
            cfg = ClusteringConfig.euclidean(n_clusters=n_clusters, random_state=seed, scale=False)
            X_train, labels, model = TimeSeriesClusterer(cfg).fit_predict(train_frame)
            inertia.append(model.inertia_)
            silhouette_coef.append(silhouette_score(X_train, labels))

        print(silhouette_coef, inertia)


if __name__ == '__main__':
    SilhouetteComparisonRunner().run()
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
