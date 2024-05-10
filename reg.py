from __future__ import print_function
from statsmodels.compat import lzip
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols
import math


use = pd.read_csv("/home/cade/Documents/Research/ABM/dynamic_pricing/py_files+data/qty_bill_all.csv")

use["log_Q"] = 2*(use["Quantity"])
#use_model = ols("Quantity ~ Billed", data=use).fit()
#print(use_model.summary())

print(use.head())
