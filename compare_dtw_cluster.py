import pandas as pd
import numpy as np
import os
import utils as ut
import time
import config
from clustering_engine import ClusteringConfig, TimeSeriesClusterer
from data import DataLoader



class CompareDtwClusterRunner:
    def __init__(self, n_clusters=4, cluster_window=1, seed=config.DEFAULT_SEED):
        self.n_clusters = n_clusters
        self.cluster_window = cluster_window
        self.seed = seed
        np.random.seed(seed)
        self.loader = DataLoader()
        self.output_path = os.path.expanduser(config.OUTPUT_PATH)
        os.makedirs(self.output_path, exist_ok=True)

    def run(self):
        df_use = self.loader.load_water_use_years(['y1_SFR_hourly.pkl', 'y2_SFR_hourly.pkl'])
        seasonal = ut.groupby_season(df_use)
        for season in ['summer', 'winter']:
            season_frame = self._slice_season(seasonal, season).T
            cfg = ClusteringConfig.dtw(
                self.n_clusters,
                radius=self.cluster_window,
                n_init=config.DEFAULT_N_INIT,
                max_iter_barycenter=config.DEFAULT_MAX_ITER_BARYCENTER,
                random_state=self.seed,
                n_jobs=-1,
            )
            _, _, model = TimeSeriesClusterer(cfg).fit_predict(season_frame)
            df = pd.DataFrame(list(zip(list(df_use.columns), model.labels_)), columns=['User', 'DBA cluster'])
            output_file = os.path.join(self.output_path, f'compare_dtw_cluster_k{self.n_clusters}_{season}.csv')
            df.to_csv(output_file, index=False)

    @staticmethod
    def _slice_season(frame, season):
        if season == 'summer':
            return frame.iloc[0:24, :]
        if season == 'autumn':
            return frame.iloc[24:48, :]
        if season == 'winter':
            return frame.iloc[48:72, :]
        return frame.iloc[72:97, :]


if __name__ == '__main__':
    begin = time.perf_counter()
    CompareDtwClusterRunner().run()
    end = time.perf_counter()
    total = (end - begin) / 60
    print(f'Clustering took {total:.2f} minutes to run')

