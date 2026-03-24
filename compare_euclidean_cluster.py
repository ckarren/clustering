import pandas as pd
import numpy as np
import os
import utils as ut
import matplotlib.pyplot as plt
import config
from clustering_engine import ClusteringConfig, TimeSeriesClusterer
from data import DataLoader
from visualizer import ClusterPlotter


class CompareEuclideanClusterRunner:
    def __init__(self, n_sample=100, n_cluster=4, seed=config.DEFAULT_SEED):
        self.n_sample = n_sample
        self.n_cluster = n_cluster
        self.seed = seed
        np.random.seed(seed)
        self.loader = DataLoader()
        self.output_path = os.path.expanduser(config.OUTPUT_PATH)
        os.makedirs(self.output_path, exist_ok=True)

    def run(self):
        use_df = self.loader.load_water_use('y1_SFR_hourly.pkl', clean=True)
        use_df = use_df.sample(n=self.n_sample, axis=1, random_state=1)
        train_frame = ut.groupby_year(use_df).T

        windows = [3, 1, 10]
        titles = ['DBA with no radius set', 'DBA with window of 1', 'DBA with window of 10']
        plt.figure()
        for row_idx, window in enumerate(windows):
            cfg = ClusteringConfig.dtw(
                self.n_cluster,
                radius=window,
                n_init=2,
                max_iter_barycenter=10,
                scale=False,
            )
            X_train, labels, model = TimeSeriesClusterer(cfg).fit_predict(train_frame)
            self._draw_row(X_train, labels, model.cluster_centers_, row_idx, titles[row_idx])

        output_file = os.path.join(self.output_path, f'compare_euclidean_cluster_windows_k{self.n_cluster}.png')
        plt.savefig(output_file)

    def _draw_row(self, X_train, labels, centers, row, row_title):
        for yi in range(self.n_cluster):
            ax = plt.subplot(3, self.n_cluster, yi + row * self.n_cluster + 1)
            ClusterPlotter.draw_cluster_panel(
                ax=ax,
                series_dataset=X_train,
                labels=labels,
                center=centers[yi],
                cluster_id=yi,
                x_limit=X_train.shape[1],
                title=row_title if yi == 1 else None,
            )


if __name__ == '__main__':
    CompareEuclideanClusterRunner().run()
