# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 15:58:32 2017

@author: jb2428

https://sites.google.com/site/aslugsguidetopython/data-analysis/pandas/calling-r-from-python

"""

from rpy2.robjects.packages import importr
import rpy2.robjects as ro
import pandas.rpy.common as com
from rpy2.robjects import pandas2ri
pandas2ri.activate()

import numpy as np
import pandas as pd
df = pd.DataFrame(np.random.normal(0,1,size=(10,3)), columns=['A','B','C'])
ro.globalenv['X'] = df
print(ro.r('summary(X)'))

ro.r('source(\'testf.R\')')
Z = ro.r('f(X)')

# https://git.yale.edu/raw/CORE-BD2K-Internal/WHI/master/ascvd_risk_calc.R?token=AAACzskEqOlkawfV4G5hNW9fYkybg-98ks5Yp3WGwA%3D%3D
ro.r('source(\'ascvd_risk_calc.R\')')
Z = ro.r('f(X)')