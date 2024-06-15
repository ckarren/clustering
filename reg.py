import numpy as np
import pandas as pd
import statsmodels.api as sm
import utils as ut

data_file = ut.load_reg_data()
data = pd.read_pickle(data_file)
data.drop(columns='P', inplace=True)
#  data.dropna(axis=0, how='any', inplace=True, ignore_index=True)

y = np.asarray(data['P_ave'])
X = np.asarray(data.loc[:, data.columns != 'logQ'])

#
#  X = np.asarray(data['P_ave'])
X = sm.add_constant(X, prepend=False)
model = sm.OLS(y, X)
results = model.fit()
results.summary().as_csv()
print(results.summary())
