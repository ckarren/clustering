import pandas as pd
import numpy as np
import utils as ut
import time
import matplotlib.pyplot as plt
from tslearn.utils import to_time_series_dataset
from tslearn.clustering import TimeSeriesKMeans, silhouette_score
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

# set variables for clustering
seed = 0
np.random.seed(seed)
n_init = 5     # number of times to initialize centroid seeds see: tslearn docs for more info
max_iter_barycenter=20      # number of iterations for the barycenter computation process see: tslearn docs for more info
cluster_window = 1       # size of the radius window when performing dynamic time warping
n_clusters = [6, 7, 8, 9]       # different values of k to interate through. Can use a list with one value of k if you don't want to compare different values of k 

file_path = str('../InputFiles/')
use_file1 = file_path + 'y1_SFR_hourly.pkl'    # hourly demand data for target population

use_df1 = pd.read_pickle(use_file1)
df_use = ut.clean_outliers(df_use)      # clean demand data of outliers
X1_train = ut.groupby_year(df_use)      # to cluster on average hourly demand over the year. to cluster on different features (montly or seasonal), see utils.py groupby_season() and groupby_month()

begin = time.perf_counter()        # to time clustering
        
# format time series data for tslearn clustering algorithm
X1_train = X1_train.T
X_train = to_time_series_dataset(X1_train)
# scale time series data so clustering is on shape and not magnitude. see tslearn docs for more info and other options for scaling
X_train = TimeSeriesScalerMeanVariance().fit_transform(X_train)

for cluster in n_clusters:
    dba_km = TimeSeriesKMeans(n_clusters=cluster,
                                  n_init=n_init,
                                  metric='dtw',
                                  max_iter_barycenter=max_iter_barycenter,
                                  random_state=seed,
                                  verbose=True,
                                  metric_params={'global_constraint':'sakoe_chiba',
                                                 'sakoe_chiba_radius':cluster_window},
                                  n_jobs=-1)

    df = pd.DataFrame(list(zip(list(df_use.columns), dba_km.labels_)),
                      columns=['User', 'DBA cluster'])
    df.to_csv(f'k{cluster}_results_dtw_r{cluster_window}.csv')

end = time.perf_counter()
total = (end - begin) / 60
print(f'Clustering took {total} minutes to run')

    
