import numpy as np
import os
import pandas as pd

import config
import utils as ut
from clustering_engine import ClusteringConfig, TimeSeriesClusterer
from data import DataLoader


class CompareDbaClustersRunner:
    def __init__(self, n_cluster=5, seed=config.DEFAULT_SEED):
        self.n_cluster = n_cluster
        self.seed = seed
        np.random.seed(seed)
        self.loader = DataLoader()
        self.output_path = os.path.expanduser(config.OUTPUT_PATH)
        os.makedirs(self.output_path, exist_ok=True)

    def run(self):
        use_df = self.loader.load_water_use('y1_SFR_hourly.pkl', clean=True)
        train_frame = ut.groupby_season(use_df).T
        cfg = ClusteringConfig.dtw(
            self.n_cluster,
            radius=2,
            n_init=config.DEFAULT_N_INIT,
            max_iter_barycenter=10,
            random_state=self.seed,
        )
        _, _, model = TimeSeriesClusterer(cfg).fit_predict(train_frame)
        df = pd.DataFrame(list(zip(list(use_df.columns), model.labels_)), columns=['User', 'DBA cluster'])
        output_file = os.path.join(self.output_path, f'compare_dba_clusters_k{self.n_cluster}.csv')
        df.to_csv(output_file, index=False)


if __name__ == '__main__':
    CompareDbaClustersRunner().run()
