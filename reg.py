import numpy as np
import os
import pandas as pd
import time
import statsmodels.api as sm
from statsmodels.sandbox.regression.gmm import IV2SLS
import utils as ut

start = time.perf_counter()
data = pd.read_pickle('B:/LAP_inst_reg_data_with_dummies.pkl')
load = time.perf_counter()
# y = np.asarray(data['Q'], dtype=np.float32)
X = np.asarray(data['P_ave'], dtype=np.float32)
inst = np.asarray(data.iloc[:,2:], dtype=np.float32)

# X = sm.add_constant(X)
inst = sm.add_constant(inst)
# model = IV2SLS(y, X, inst)

model = sm.OLS(X, inst)
results = model.fit()
print(results.summary())
end = time.time()
end = time.perf_counter()
# results.summary().as_csv()
print('load time: ', load - start)
print('total time: ', end - start)
