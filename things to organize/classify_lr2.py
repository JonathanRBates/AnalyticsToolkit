# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 10:29:41 2017

@author: jb2428
"""

from __future__ import print_function

import sys
import pickle as pk
# import cPickle as pk
from analytic_toolkit import preprocess_train_data

##############################################################################

print(sys.version)

##############################################################################
# LOAD DATA
# cf create_reference_standard
##############################################################################

X_train, X_test, y_train, y_test, tr = pk.load(open(DATA, 'rb'))

print('Train count:\n', y_train.value_counts())
print('Test count:\n', y_test.value_counts())

Xt, mapper, column_names = preprocess_train_data(X_train, tr)

import statsmodels.api as sm
import pandas as pd
Xt = pd.DataFrame(Xt, columns=column_names, index=X_train.index)
Xt['intercept'] = 1.0
logit = sm.Logit(y_train, Xt)
result = logit.fit()
print(result.summary())

# cf http://statsmodels.sourceforge.net/devel/generated/statsmodels.iolib.summary.Summary.html

with open(r'lr.html','w') as f:
    f.write(result.summary().as_html())
    
# from statsmodels import table
# table.SimpleTable(result.summary())