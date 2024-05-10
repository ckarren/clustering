#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import utils as ut
from tslearn.utils import to_time_series_dataset
from tslearn.clustering import TimeSeriesKMeans, silhouette_score
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, \
    TimeSeriesResampler
from tslearn.datasets import CachedDatasets
import plotly.graph_objects as go
from plotly.subplots import make_subplots

seed = 0
np.random.seed(seed)
n_sample = 100

file_path = str('~/OneDrive - North Carolina State University/Documents/Clustering+Elasticity/InputFiles/')
use_file = file_path + 'y1_SFR_hourly.pkl'

use_df = pd.read_pickle(use_file)
use_df = ut.clean_outliers(use_df)
use_df = use_df.sample(n=n_sample, axis=1, random_state=1)
user_df = pd.DataFrame(use_df.columns)
user_df.to_csv('100users.csv')
