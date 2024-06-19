import numpy as np
import pandas as pd
import time
import statsmodels.api as sm
#  from statsmodels.sandbox.regression.gmm import IV2SLS
import utils as ut

start = time.time()
#  data_file = ut.load_reg_data()
data = pd.read_pickle('B:/LAP_inst_reg_data_with_dummies.pkl')
load = time.time()
col_names = data.columns.tolist()
np.savetxt('columns_reg.csv', col_names, delimiter=",", fmt='% s')
#  y = np.asarray(data['Q'], dtype=np.float32)
X = np.asarray(data['P_ave'], dtype=np.float32)
inst = np.asarray(data.iloc[:,2:], dtype=np.float32)

#  X = sm.add_constant(X)
inst = sm.add_constant(inst)
#  model = IV2SLS(y, X, inst)
#  data.to_stata('lap_inst_data.dta')

model = sm.OLS(X, inst)
results = model.fit()
end = time.time()
hat_ln_price = model.predict(params=inst[1:])
print(hat_ln_price)
results.summary().as_csv()
print(results.summary())
print('load time: ', load - start)
print('total time: ', end - start)
