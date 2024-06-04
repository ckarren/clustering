from __future__ import print_function
from statsmodels.compat import lzip
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols
import math


data = pd.read_pickle('reg_data_500_with_dummies.pkl')
y = np.asarray(data['logQ'])
#  X = np.asarray(data.loc[:, data.columns != 'logQ'])
X = np.asarray(data['logP'])
X = sm.add_constant(X)
model = sm.OLS(y, X)
results = model.fit()

