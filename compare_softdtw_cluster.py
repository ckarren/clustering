import pandas as pd
import numpy as np
import os
import utils as ut
import matplotlib.pyplot as plt
import config
from clustering_engine import ClusteringConfig, TimeSeriesClusterer
from data import DataLoader
from visualizer import ClusterPlotter



class CompareSoftDtwClusterRunner:
    def __init__(self, n_cluster=5, seed=config.DEFAULT_SEED):
        self.n_cluster = n_cluster
        self.seed = seed
        np.random.seed(seed)
        self.loader = DataLoader()
        self.output_path = os.path.expanduser(config.OUTPUT_PATH)
        os.makedirs(self.output_path, exist_ok=True)

    def run(self):
        use_df = self.loader.load_water_use('y1_SFR_hourly.pkl', clean=True)
        train_frame = ut.groupby_year(use_df).T

        e_cfg = ClusteringConfig.euclidean(self.n_cluster)
        X_train, e_labels, e_model = TimeSeriesClusterer(e_cfg).fit_predict(train_frame)

        d_cfg = ClusteringConfig.dtw(self.n_cluster, radius=1)
        _, d_labels, d_model = TimeSeriesClusterer(d_cfg).fit_predict(train_frame)

        self._plot(X_train, e_labels, e_model.cluster_centers_, d_labels, d_model.cluster_centers_)
        self._write_labels(use_df, e_model.labels_, d_model.labels_)

    def _plot(self, X_train, e_labels, e_centers, d_labels, d_centers):
        plt.figure()
        for yi in range(self.n_cluster):
            ax = plt.subplot(2, self.n_cluster, yi + 1)
            ClusterPlotter.draw_cluster_panel(
                ax=ax,
                series_dataset=X_train,
                labels=e_labels,
                center=e_centers[yi],
                cluster_id=yi,
                x_limit=X_train.shape[1],
                title='Euclidean' if yi == 2 else None,
            )

        for yi in range(self.n_cluster):
            ax = plt.subplot(2, self.n_cluster, yi + self.n_cluster + 1)
            ClusterPlotter.draw_cluster_panel(
                ax=ax,
                series_dataset=X_train,
                labels=d_labels,
                center=d_centers[yi],
                cluster_id=yi,
                x_limit=X_train.shape[1],
                y_limit=(-4, 4),
                title='DBA with window of 1' if yi == 2 else None,
            )
        output_file = os.path.join(self.output_path, f'compare_softdtw_cluster_plot_k{self.n_cluster}.png')
        plt.savefig(output_file)

    def _write_labels(self, use_df, euclidean_labels, dba_labels):
        df = pd.DataFrame(
            list(zip(list(use_df.columns), euclidean_labels, dba_labels)),
            columns=['User', 'k-means cluster', 'DBA cluster'],
        )
        output_file = os.path.join(self.output_path, f'compare_softdtw_cluster_labels_k{self.n_cluster}.csv')
        df.to_csv(output_file, index=False)


if __name__ == '__main__':
    CompareSoftDtwClusterRunner().run()
