import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from tslearn.clustering import KernelKMeans

import config
import utils as ut
from clustering_engine import ClusteringConfig, TimeSeriesClusterer
from data import DataLoader
from visualizer import ClusterPlotter


class ClusterExampleRunner:
    def __init__(self, n_clusters=4, seed=config.DEFAULT_SEED):
        self.n_clusters = n_clusters
        self.seed = seed
        np.random.seed(seed)
        self.loader = DataLoader()
        self.output_path = os.path.expanduser(config.OUTPUT_PATH)
        os.makedirs(self.output_path, exist_ok=True)

    def _training_frame(self):
        use_df = self.loader.load_water_use('y1_SFR_hourly.pkl', clean=True)
        return use_df, ut.groupby_year(use_df).T

    def run(self):
        use_df, train_frame = self._training_frame()
        ClusterPlotter.new_figure(rows=4, cols=self.n_clusters)

        e_cfg = ClusteringConfig.euclidean(self.n_clusters, random_state=self.seed, scale=False)
        X_train, e_labels, e_model = TimeSeriesClusterer(e_cfg).fit_predict(train_frame)
        self._draw_row(X_train, e_labels, e_model.cluster_centers_, 0, 'Euclidean $k$-means', y_limit=(0, 55))

        d_cfg = ClusteringConfig.dtw(
            self.n_clusters,
            random_state=self.seed,
            n_init=2,
            max_iter_barycenter=20,
            metric_params={},
            scale=False,
        )
        _, d_labels, d_model = TimeSeriesClusterer(d_cfg).fit_predict(train_frame)
        self._draw_row(X_train, d_labels, d_model.cluster_centers_, 1, 'DBA $k$-means')

        s_cfg = ClusteringConfig.softdtw(
            self.n_clusters,
            random_state=self.seed,
            scale=False,
            metric_params={},
            gamma=0.01,
        )
        _, s_labels, s_model = TimeSeriesClusterer(s_cfg).fit_predict(train_frame)
        self._draw_row(X_train, s_labels, s_model.cluster_centers_, 2, 'Soft-DTW $k$-means')

        k_model = KernelKMeans(
            n_clusters=self.n_clusters,
            kernel='gak',
            verbose=True,
            random_state=self.seed,
            kernel_params={'sigma': 'auto'},
            n_init=2,
        )
        k_labels = k_model.fit_predict(X_train)
        self._draw_row(X_train, k_labels, [None] * self.n_clusters, 3, 'Kernel $k$-means')

        self._write_results(use_df, e_model.labels_, d_model.labels_, s_model.labels_, k_model.labels_)
        plt.tight_layout()
        output_file = os.path.join(self.output_path, f'cluster_example_compare_metrics_k{self.n_clusters}.png')
        plt.savefig(output_file)

    def _draw_row(self, X_train, labels, centers, row, title, y_limit=None):
        x_limit = X_train.shape[1]
        for yi in range(self.n_clusters):
            ax = plt.subplot(4, self.n_clusters, yi + (row * self.n_clusters) + 1)
            center = centers[yi] if centers[yi] is not None else None
            ClusterPlotter.draw_cluster_panel(
                ax=ax,
                series_dataset=X_train,
                labels=labels,
                center=center,
                cluster_id=yi,
                x_limit=x_limit,
                y_limit=y_limit,
                title=title if yi == 1 else None,
            )

    def _write_results(self, use_df, euclidean_labels, dba_labels, softdtw_labels, kernel_labels):
        df = pd.DataFrame(
            list(zip(list(use_df.columns), euclidean_labels, dba_labels, softdtw_labels, kernel_labels)),
            columns=['User', 'k-means cluster', 'DBA cluster', 'SoftDTW cluster', 'Kernel k-means cluster'],
        )
        output_file = os.path.join(self.output_path, f'cluster_example_labels_k{self.n_clusters}.csv')
        df.to_csv(output_file, index=False)


if __name__ == '__main__':
    ClusterExampleRunner().run()
