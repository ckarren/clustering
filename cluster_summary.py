#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import utils as ut
from utils import cluster_lot

n_clusters = 5
radius = 1
fontsize = 18

lot_df = cluster_lot(n_clusters=n_clusters, radius=radius)
#  atts = ['EffectiveYearBuilt', 'SQFTmain', 'Bedrooms', 'Bathrooms', 'TotalValue']
cluster_file = f'../RadiusComps/{n_clusters}_DTW_results_scaled_r{radius}.csv'
cluster_data = pd.read_csv(cluster_file, 
                           usecols=[1,2], 
                           index_col=0)
cluster_df = pd.DataFrame(cluster_data)

#  df_use1 = pd.read_pickle('../InputFiles/y1_SFR_hourly.pkl')
#  df_use2 = pd.read_pickle('../InputFiles/y2_SFR_hourly.pkl')
#  df_use = pd.concat([df_use1, df_use2], join='inner')
#  df_use = ut.clean_outliers(df_use)
#  df_use = df_use.T
#  df_use = df_use.join(cluster_df, how='inner')
#  lot_summary_stat = lot_df[atts].describe()
#  lot_summary_stat.to_csv('all_summary_stats.csv')
lot_df = cluster_lot(n_clusters=n_clusters, radius=radius)
#  atts = ['EffectiveYearBuilt', 'SQFTmain', 'Bedrooms', 'Bathrooms', 'TotalValue']

#  lot_summary_stat = lot_df[atts].describe()
#  lot_summary_stat.to_csv('all_summary_stats.csv')

#  for att in atts:
    #  fig, axs = plt.subplots(1, n_clusters, tight_layout=True, sharey=True)
    #  for i in range(n_clusters):
        #  cluster_lot = lot_df.loc[lot_df['DBA cluster'] == i]
        #  axs[i].hist(cluster_lot[att], bins=5, density=True, histtype='bar')
        #  axs[i].set_title(f'Cluster {i+1}')
    #  fig.supxlabel(f'{att}')
    #  fig.supylabel('Probability density')
    #  plt.show()

grouped = lot_df.groupby('DBA cluster')[['EffectiveYearBuilt',
                                         'SQFTmain',
                                        'Bedrooms', 
                                         'Bathrooms',
                                        'TotalValue']].agg(['median'])
grouped.to_csv('5_clusters_summary_stats_median.csv')
