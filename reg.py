import numpy as np
import pandas as pd
import statsmodels.api as sm
import utils as ut 
import os

n_sample = 250

data_file = f'reg_data_{n_sample}_with_dummies.pkl'
if os.path.exists(data_file):
    data_file = data_file
else:
    ut.prepare_regression(sample=True, n_sample=n_sample)
    ut.add_dummies('reg_data_' + str(n_sample) + '.pkl')
    data_file = data_file
#  data_file = 'reg_data_with_dummies.pkl'
data = pd.read_pickle(data_file)
data.dropna(axis=0, how='any', inplace=True, ignore_index=True)
for i in data:
    s = data[i].isna().sum()
    if s != 0:
        print(s)
y = np.asarray(data['logQ']  )
#  X = np.asarray(data.loc[:, data.columns != 'logQ'])
#
X = np.asarray(data['logP'])
X = sm.add_constant(X, prepend=False)
model = sm.OLS(y, X, missing='raise')
results = model.fit()
print(results.summary())

