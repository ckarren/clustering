#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import matplotlib.pyplot as plt
#  import plotly.express as px
#  import plotly.graph_objects as go
import config
import utils as ut 
from data import DataLoader
fontsize = 12
tfontsize = 14


class DTWClusterAnalysisRunner:
    def __init__(self):
        self.loader = DataLoader()
        self.output_path = os.path.expanduser(config.OUTPUT_PATH)
        os.makedirs(self.output_path, exist_ok=True)

    def run(self):
        df_use = self.loader.load_water_use('y1_SFR_hourly.pkl', clean=False)
        df_use_y = ut.groupby_year(df_use)
        for n_cluster in [4, 5, 6]:
            self._plot_cluster_means(df_use_y, n_cluster)

    def _plot_cluster_means(self, df_use_y, n_cluster):
        df = ut.analyse_dtw(n_cluster)
        radius = df['r1']
        clusters = list(range(n_cluster))
        fig, axs = plt.subplots(1, n_cluster, figsize=(48, 12), sharex=True, sharey=True, layout='constrained')
        for ci, c in enumerate(clusters):
            members = [str(x) for x in radius[radius == c].index.to_list()]
            df_use_rc = df_use_y.filter(items=members)
            average = df_use_rc.mean(axis=1)
            peak_hour = str(average.idxmax())
            axs[ci].plot(df_use_rc.index, average, c='crimson', linewidth=3)
            axs[ci].annotate(f'peak hour: {peak_hour}', xy=(12, 4))

        fig.supxlabel('Time (hr)', fontsize=fontsize)
        fig.supylabel('Volume (gallons)', fontsize=fontsize)
        fig.suptitle(f'Cluster Averages for k={n_cluster}', fontsize=tfontsize)
        output_file = os.path.join(self.output_path, f'dtw_cluster_analysis_means_k{n_cluster}.png')
        plt.savefig(output_file)


if __name__ == '__main__':
    DTWClusterAnalysisRunner().run()
