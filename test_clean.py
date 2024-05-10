#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd 
import utils as ut

input_path = str('~/OneDrive - North Carolina State University/Documents/Clustering+Elasticity/InputFiles/')
df = pd.read_pickle(input_path + 'y1_SFR_hourly.pkl')
#  df = df.drop('1548338250', axis=1)
a = ut.summary(df)
df = ut.clean_outliers(df)
b = ut.summary(df)
print(a, b)
#  for i in df['1547533344']:
    #  if i > 400:
        #  print(i)
