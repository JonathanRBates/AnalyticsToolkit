# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 11:05:22 2017

@author: jb2428

Note: statsmodels vs scikit learn: 

    http://stats.stackexchange.com/questions/203740/logistic-regression-scikit-learn-vs-statsmodels
    
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


N = 100000


X = pd.DataFrame(np.random.random((N, 2)),columns=['x1','x2'])
X = sm.add_constant(X)
noise = np.random.random(N)
beta = [1., .1, .5]
y = np.dot(X, beta) + 0.1*noise


# Fit regression model
results = sm.OLS(y, X).fit()

# Inspect the results
print(results.summary())





