#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd 
import glob
import matplotlib.pyplot as plt
import plotly.express as px 
import plotly.graph_objects as go 
import utils as ut 

file = (str('./InputFiles/y1_SFR_hourly.pkl'))

df_use = pd.read_pickle(file)
df_use_y = ut.groupby_year(df_use)
df_use_m = ut.groupby_month(df_use)
df = ut.analyse_dtw(4)

r1 = df['r1']
r1_cluster0 = [str(x) for x in r1[r1 == 3].index.to_list()]
df_use_r1c0 =  df_use_y.filter(items=r1_cluster0)
r1c0_total = df_use_r1c0.sum(axis=1)
r1c0_average = df_use_r1c0.mean(axis=1)

fig, ax = plt.subplots()
for i in df_use_r1c0.columns:
    i = str(i)
    ax.plot(df_use_r1c0.index, df_use_r1c0[i], c='grey')
ax.set_title('Average hourly use for Cluster 4', fontsize=14)
ax.set_xlabel('Time (hr)', fontsize=14)
ax.set_ylabel('Volume (gallons)', fontsize=14)
ax.plot(df_use_r1c0.index, r1c0_average, c='crimson')
plt.show()

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
