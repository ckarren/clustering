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
df_use = ut.clean_outliers(df_use)
df_use_y = ut.groupby_year(df_use)
df_use_m = ut.groupby_month(df_use)
#  fig, ax = plt.subplots(
        #  figsize=(24, 6),
        #  layout='constrained')
total = df_use_y.sum(axis=1)
average = df_use_y.mean(axis=1)
total.to_csv('daily_total.csv')
average.to_csv('daily_average.csv')
        #  for i in df_use_rc.columns:
            #  i = str(i)
            #  ax.plot(df_use_rc.index, df_use_rc[i], c='grey')
#  ax.plot(df_use_y.index, total, c='crimson')
        #  ax[ci].set_title(f'Average Use for Cluster {c}, radius {r}', fontsize=14)
        #  axs[ci].set_xlabel('Time (hr)', fontsize=fontsize)
        #  axs[ci].set_ylabel('Volume (gallons)', fontsize=fontsize)
#  ax.set_xlabel('Time (hr)', fontsize=fontsize)
#  ax.set_ylabel('Volume (gallons)', fontsize=fontsize)
#  plt.show()


    #  plt.savefig(f'../{n_clusters}_clusters-means_{r}.png')

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