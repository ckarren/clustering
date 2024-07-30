#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
#  from utils import cluster_lot

n_clusters = 5
radius = 1
fontsize = 18

#  lot_df = cluster_lot(n_clusters=n_clusters, radius=radius)
#  atts = ['EffectiveYearBuilt', 'SQFTmain', 'Bedrooms', 'Bathrooms', 'TotalValue']
use_bill = pd.read_pickle('LAP_inst_reg_data.pkl')
print(use_bill.head())
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

#  grouped = lot_df.groupby('DBA cluster')[['EffectiveYearBuilt', 'SQFTmain',
                                        #  'Bedrooms', 'Bathrooms',
                                        #  'TotalValue']].agg(['count', 'max',
                                                           #  'min', 'mean',
                                                           #  'std'])
#  grouped.to_csv('5_clusters_summary_stats.csv')
