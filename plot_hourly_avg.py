#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
#  import plotly.express as px
#  import plotly.graph_objects as go
import utils as ut 
from data import DataLoader
fontsize = 12


class HourlyAveragePlotter:
    def __init__(self):
        self.loader = DataLoader()

    def run(self):
        df_use = self.loader.load_water_use('y1_SFR_hourly.pkl', clean=True).multiply(7.48)
        df_use_y = ut.groupby_year(df_use)
        average = df_use_y.mean(axis=1).multiply(7.48)

        fig, ax = plt.subplots()
        ax.plot(df_use_y.index, average, c='darkblue', marker='o', linewidth=3)
        ax.set_xlabel('Time (hr)', fontsize=fontsize)
        ax.set_ylabel('Volume (gallons)', fontsize=fontsize)
        plt.show()


if __name__ == '__main__':
    HourlyAveragePlotter().run()
    #  plt.savefig(f'../{n_clusters}_clusters-means_{r}.png')
