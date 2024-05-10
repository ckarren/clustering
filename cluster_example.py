import numpy
import pandas as pd
import utils as ut
import matplotlib.pyplot as plt
from tslearn.utils import to_time_series_dataset
from tslearn.clustering import TimeSeriesKMeans, KernelKMeans, silhouette_score
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

seed = 0
n_clusters = 4
numpy.random.seed(seed)

file_path = str('~/OneDrive - North Carolina State University/Documents/Clustering+Elasticity/InputFiles/')
use_file = file_path + 'y1_SFR_hourly.pkl'

use_df = pd.read_pickle(use_file)
use_df = ut.clean_outliers(use_df)
#  use_df = use_df.sample(1800, axis=1, random_state=1)
X_train = ut.groupby_year(use_df)
X_train = X_train.T
X_train = to_time_series_dataset(X_train)
#  X_train = TimeSeriesScalerMeanVariance().fit_transform(X_train)
sz = X_train.shape[1]

# Euclidean k-means
print("Euclidean k-means")
km = TimeSeriesKMeans(n_clusters=n_clusters, 
                      verbose=True, 
                      random_state=seed)
y_pred = km.fit_predict(X_train)

plt.figure()
for yi in range(n_clusters):
    plt.subplot(4, n_clusters, yi + 1)
    for xx in X_train[y_pred == yi]:
        plt.plot(xx.ravel(), "k-", alpha=.2)
    plt.plot(km.cluster_centers_[yi].ravel(), "r-")
    plt.xlim(0, sz)
    plt.ylim(0,55)
    plt.text(0.55, 0.85,'Cluster %d' % (yi + 1),
             transform=plt.gca().transAxes)
    if yi == 1:
        plt.title("Euclidean $k$-means")

# DBA-k-means
print("DBA k-means")
dba_km = TimeSeriesKMeans(n_clusters=n_clusters,
                          n_init=2,
                          metric="dtw",
                          verbose=True,
                          max_iter_barycenter=20,
                          random_state=seed)
y_pred = dba_km.fit_predict(X_train)
#
for yi in range(n_clusters):
    plt.subplot(4, n_clusters, yi + n_clusters + 1)
    for xx in X_train[y_pred == yi]:
        plt.plot(xx.ravel(), "k-", alpha=.2)
    plt.plot(dba_km.cluster_centers_[yi].ravel(), "r-")
    plt.xlim(0, sz)
    plt.text(0.55, 0.85,'Cluster %d' % (yi + 1),
             transform=plt.gca().transAxes)
    if yi == 1:
        plt.title("DBA $k$-means")
#
#  Soft-DTW-k-means
print("Soft-DTW k-means")
sdtw_km = TimeSeriesKMeans(n_clusters=n_clusters,
                           metric="softdtw",
                           metric_params={"gamma": .01},
                           verbose=True,
                           random_state=seed)
y_pred = sdtw_km.fit_predict(X_train)
#
for yi in range(n_clusters):
    plt.subplot(4, n_clusters, yi + 2*n_clusters + 1)
    for xx in X_train[y_pred == yi]:
        plt.plot(xx.ravel(), "k-", alpha=.2)
    plt.plot(sdtw_km.cluster_centers_[yi].ravel(), "r-")
    plt.xlim(0, sz)
    plt.text(0.55, 0.85,'Cluster %d' % (yi + 1),
             transform=plt.gca().transAxes)
    if yi == 1:
        plt.title("Soft-DTW $k$-means")

# Kernel k-means
print("Kernel k-means")
kernel_km = KernelKMeans(n_clusters=n_clusters, 
                         kernel='gak', 
                         verbose=True, 
                         random_state=seed,
                         kernel_params={'sigma': 'auto'},
                         n_init=2)
y_pred = kernel_km.fit_predict(X_train)

for yi in range(n_clusters):
    plt.subplot(4, n_clusters, yi + 3*n_clusters + 1)
    for xx in X_train[y_pred == yi]:
        plt.plot(xx.ravel(), "k-", alpha=.2)
    plt.xlim(0, sz)
    plt.text(0.55, 0.85,'Cluster %d' % (yi + 1),
             transform=plt.gca().transAxes)
    if yi == 1:
        plt.title("Kernel $k$-means")

df = pd.DataFrame(list(zip(list(use_df.columns), 
                           km.labels_, 
                           dba_km.labels_,
                           sdtw_km.labels_,
                           kernel_km.labels_)),
                  columns=['User', 'k-means cluster', 'DBA cluster', 
                           'SoftDTW cluster', 'Kernel k-means cluster'])
df.to_csv(f'{n_clusters}_compare_n1800_year.csv')

plt.tight_layout()
plt.savefig(f'compare_metrics_{n_clusters}_n1800_season.png')
#  plt.show()
