#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd 
#  import glob
import matplotlib.pyplot as plt
#  import plotly.express as px
#  import plotly.graph_objects as go
import utils as ut 
fontsize = 12
tfontsize = 14
file = (str('../InputFiles/y1_SFR_hourly.pkl'))

df_use = pd.read_pickle(file)
df_use_y = ut.groupby_year(df_use)
#  df_use_m = ut.groupby_month(df_use)
n_clusters = [4,5,6]
radii = ['r1']

for n_cluster in n_clusters:
    df = ut.analyse_dtw(n_cluster)
    radius = df['r1']
    clusters = list(range(n_cluster))
    fig, axs = plt.subplots(
        1, n_cluster, 
        figsize=(48, 12), 
        sharex=True,
        sharey=True,
        layout='constrained')
    for ci, c in enumerate(clusters):
        cluster = [str(x) for x in radius[radius == c].index.to_list()]
        df_use_rc =  df_use_y.filter(items=cluster)
        total = df_use_rc.sum(axis=1)
        average = df_use_rc.mean(axis=1)
        peak_hour = str(average.idxmax())

        axs[ci].plot(df_use_rc.index, average, c='crimson', linewidth=3)
        axs[ci].annotate(f'peak hour: {peak_hour}', 
                         xy=(12,4)
                         )
        #  ax[ci].set_title(f'Average Use for Cluster {c}, radius {r}', fontsize=14)
        #  axs[ci].set_xlabel('Time (hr)', fontsize=fontsize)
        #  axs[ci].set_ylabel('Volume (gallons)', fontsize=fontsize)
    fig.supxlabel('Time (hr)', fontsize=fontsize)
    fig.supylabel('Volume (gallons)', fontsize=fontsize)
    fig.suptitle(f'Cluster Averages for k={n_cluster}', fontsize=tfontsize)
    plt.show()
    plt.savefig(f'../{n_clusters}_clusters-means_peaks.png')
