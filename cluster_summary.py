#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

import config
import pandas as pd
import numpy as np

#  atts = ['EffectiveYearBuilt', 'SQFTmain', 'Bedrooms', 'Bathrooms', 'TotalValue']

#  df_use1 = pd.read_pickle('../InputFiles/y1_SFR_hourly.pkl')
#  df_use2 = pd.read_pickle('../InputFiles/y2_SFR_hourly.pkl')
#  df_use = pd.concat([df_use1, df_use2], join='inner')
#  df_use = ut.clean_outliers(df_use)
#  df_use = df_use.T
#  df_use = df_use.join(cluster_df, how='inner')
#  lot_summary_stat = lot_df[atts].describe()
#  lot_summary_stat.to_csv('all_summary_stats.csv')
#  lot_df = cluster_lot(n_clusters=n_clusters, radius=radius)
lot_file = pd.read_csv(os.path.join(os.path.expanduser(config.OUTPUT_PATH), 'cluster_lot_info_parcel_size.csv'),
                       usecols=[1,5,7,8,9,10,14,19,21])
lot_df = pd.DataFrame(lot_file)
lot_df['SQFTmain'] = lot_df['SQFTmain'].replace(0, np.NaN)
lot_df.dropna(subset=['SQFTmain'])
lot_df['SpecificUseDetail2'] = lot_df['SpecificUseDetail2'].map({'Pool':1,
                                                                 'Pool and Misc.':1}).fillna(0)
#  atts = ['EffectiveYearBuilt', 'SQFTmain', 'Bedrooms', 'Bathrooms', 'TotalValue']

#  lot_summary_stat = lot_df[atts].describe()
#  lot_summary_stat = lot_df[atts].agg(['median'])
#  lot_summary_stat.to_csv('all_median_stats.csv')

#  for att in atts:

    #  fig, axs = plt.subplots(1, n_clusters, tight_layout=True, sharey=True)
    #  for i in range(n_clusters):
        #  cluster_lot = lot_df.loc[lot_df['DBA cluster'] == i]
        #  axs[i].hist(cluster_lot[att], bins=5, density=True, histtype='bar')
        #  axs[i].set_title(f'Cluster {i+1}')
    #  fig.supxlabel(f'{att}')
    #  fig.supylabel('Probability density')
    #  plt.show()



class ClusterSummaryRunner:
    def __init__(self):
        self.lot_df = lot_df
        self.output_path = os.path.expanduser(config.OUTPUT_PATH)
        os.makedirs(self.output_path, exist_ok=True)

    def run(self):
        grouped = self.lot_df.groupby('DBA cluster')[[
            'EffectiveYearBuilt',
            'SQFTmain',
            'Bedrooms',
            'Bathrooms',
            'TotalValue',
            'Shape_Area',
        ]].agg(['max', 'min', 'mean', 'median'])
        output_file = os.path.join(self.output_path, 'cluster_summary_stats_parcel_median.csv')
        grouped.to_csv(output_file)


if __name__ == '__main__':
    ClusterSummaryRunner().run()
