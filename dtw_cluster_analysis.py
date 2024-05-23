#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd 
#  import glob
import matplotlib.pyplot as plt
#  import plotly.express as px
#  import plotly.graph_objects as go
import utils as ut 

file = (str('../InputFiles/y1_SFR_hourly.pkl'))

df_use = pd.read_pickle(file)
df_use_y = ut.groupby_year(df_use)
df_use_m = ut.groupby_month(df_use)
n_clusters = 4
df = ut.analyse_dtw(n_clusters)
radii = ['r1', 'r2', 'r3', 'r4', 'r5']
clusters = list(range(n_clusters))

for r in radii:
    radius = df[r]
    for c in clusters:
        cluster = [str(x) for x in radius[radius == c].index.to_list()]
        df_use_rc =  df_use_y.filter(items=cluster)
        total = df_use_rc.sum(axis=1)
        average = df_use_rc.mean(axis=1)

        fig, ax = plt.subplots()
        #  for i in df_use_rc.columns:
            #  i = str(i)
            #  ax.plot(df_use_rc.index, df_use_rc[i], c='grey')
        ax.set_title(f'Average Use for Cluster {c}, radius {r}', fontsize=14)
        ax.set_xlabel('Time (hr)', fontsize=14)
        ax.set_ylabel('Volume (gallons)', fontsize=14)
        ax.plot(df_use_rc.index, average, c='crimson')
        plt.savefig(f'../{n_clusters}_clusters-means_r{r}c{c}_average.png')

# fig = go.Figure(data=[go.Scatter(x=df_use_r1c0.index, y=r1c0_average)])

# for i in df_use_r1c0.columns:
#     i = str(i)
#     if len(df_use_r1c0[i]) == len(df_use_r1c0):
#         fig.add_trace(go.Scatter(x=df_use_r1c0.index, y=df_use_r1c0[i],
#         mode='lines',
#         name=i))
#     else:
#         pass
# fig.show()
