import numpy as np
import os
import pandas as pd
from tslearn.clustering import silhouette_score

import config
import utils as ut
from clustering_engine import ClusteringConfig, TimeSeriesClusterer
from data import DataLoader


class ClusterCompareMetricsRunner:
    def __init__(self, seed=config.DEFAULT_SEED):
        self.seed = seed
        np.random.seed(seed)
        self.loader = DataLoader()
        self.n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.groups = ['year', 'season', 'month']
        self.output_path = os.path.expanduser(config.OUTPUT_PATH)
        os.makedirs(self.output_path, exist_ok=True)

    def run(self):
        use_df = self.loader.load_water_use('y1_SFR_hourly.pkl', clean=True)
        metrics_rows = []
        for group in self.groups:
            grouped = self._group(use_df, group).T
            for n_cluster in self.n_clusters:
                cfg = ClusteringConfig.dtw(
                    n_cluster,
                    radius=2,
                    n_init=config.DEFAULT_N_INIT,
                    max_iter_barycenter=config.DEFAULT_MAX_ITER_BARYCENTER,
                    random_state=self.seed,
                )
                X_train, labels, model = TimeSeriesClusterer(cfg).fit_predict(grouped)
                sil = silhouette_score(X_train, labels)
                metrics_rows.append({
                    'group': group,
                    'n_cluster': n_cluster,
                    'silhouette_score': sil,
                    'inertia': model.inertia_,
                })

                out = pd.DataFrame(list(zip(list(use_df.columns), model.labels_)), columns=['User', 'DBA cluster'])
                labels_file = os.path.join(self.output_path, f'cluster_compare_metrics_{group}_k{n_cluster}.csv')
                out.to_csv(labels_file, index=False)

            summary_file = os.path.join(self.output_path, 'cluster_compare_metrics_summary.csv')
            pd.DataFrame(metrics_rows).to_csv(summary_file, index=False)

    @staticmethod
    def _group(use_df, group):
        if group == 'year':
            return ut.groupby_year(use_df)
        if group == 'season':
            return ut.groupby_season(use_df)
        return ut.groupby_month(use_df)


if __name__ == '__main__':
    ClusterCompareMetricsRunner().run()
