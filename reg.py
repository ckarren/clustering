import numpy as np
import os
import pandas as pd
import time
import statsmodels.api as sm
from linearmodels.iv import IV2SLS as LM2SLS 
from statsmodels.sandbox.regression.gmm import IV2SLS as SM2SLS
from statsmodels.stats.outliers_influence import variance_inflation_factor
from utils import prep_ols, prep_lm_2sls, prep_sm_2sls

start = time.perf_counter()
data = pd.read_pickle('LAP_inst_reg_data_100_with_dummies.pkl')
load = time.perf_counter()

y, X1, X, inst = prep_lm_2sls(data)
model = LM2SLS(y, X1, X, inst)
results = model.fit()
print(results.summary)
end = time.time()
end = time.perf_counter()
#  with open('lm_IV2sls_results.csv', 'w') as fh:
    #  fh.write(results.summary().as_csv())

print('load time: ', load - start)
print('total time: ', end - start)
