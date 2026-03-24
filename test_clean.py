#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd 
import utils as ut
from data import DataLoader


class CleanOutlierCheck:
    def __init__(self):
        self.loader = DataLoader()

    def run(self):
        df = self.loader.load_water_use('y1_SFR_hourly.pkl', clean=False)
        before = ut.summary(df)
        after = ut.summary(ut.clean_outliers(df))
        print(before, after)


if __name__ == '__main__':
    CleanOutlierCheck().run()
#  for i in df['1547533344']:
    #  if i > 400:
        #  print(i)
