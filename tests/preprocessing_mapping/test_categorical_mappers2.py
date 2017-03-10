# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 20:58:42 2017

@author: jb2428
"""

# -*- coding: utf-8 -*-
"""
Note, I am not using .to_categorical method for pandas dataframes,
since this does not easily allow for fitting to training set,
then applying the fit transformation to a new test set -
especially in the setting of cross-validation.


Conclusion from example below is that Method 3,

        LabelBinarizer().fit_transform(data.astype(str))
    
gives desired performance.


See 
http://stackoverflow.com/questions/21057621/sklearn-labelencoder-with-never-seen-before-values

Created on Sun Jan 15 13:25:54 2017

@author: jb2428
"""


import numpy as np
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, OneHotEncoder


if False:                       # Methods work, but difference in output number of dimensions
    t = np.array(['a','a','A','A','a'])
    ss = [np.array(['a','a','a']),
         np.array(['A','A','A']),
         np.array(['a','A','b'])]   
                     
    
if False:
    t = np.array([0,0,0,0])   # Methods are equivalent
    ss = [np.array([0,0,0]),
         np.array([1,1,0])] 

if False:
    t = np.array([np.nan,np.nan,np.nan,np.nan])   # Methods are equivalent
    ss = [np.array([0,0,0]),
         np.array([np.nan,1,0])] 
 
if False:
    t = np.array([0,3,0,3,0,0,np.nan,0,np.nan])   # Method 3 is optimal
    ss = [np.array([0,3,3.])]
          
if True:   
    t = np.array(['a',np.nan,np.nan,'a',34.,34.])  # Methods are equivalent
    ss = [np.array([0,0,'a'])]




#==============================================================================
# # Method 1 -- CANT HANDLE NEW DATA IN TRANSFORM
# print('Method 1: LabelEncoder+OneHotEncoder')
# u = LabelEncoder()
# v = OneHotEncoder(sparse=False)
# x = u.fit_transform(t)
# y = v.fit_transform(x.reshape(-1,1))
# for i, a in enumerate(t):
#     print(a, ' : ', y[i,:])
# 
# print('testing transform...')
# for s in ss:
#     print('.')
#     z = v.transform(u.transform(s).reshape(-1,1))
#     for i, a in enumerate(s):
#         print(a, ' : ', z[i,:])
#==============================================================================
   
    
# Method 3
print('Method 3: LabelBinarizer with type casting')

from sklearn.base import BaseEstimator, TransformerMixin
class MapToStr(BaseEstimator, TransformerMixin):
    """Convert data to str datatype in scikit learn Pipeline.
    
    This is created as a workaround to errors with LabelBinarizer()
    when np.nan is in the data.
    
    See test_categorical.py for motivation behind this.    
    
    Example.
    t = np.array(['a',np.nan,np.nan,'a',34.,34.])
    h = MapToStr()
    h.fit(t)         # this returns self, which is MapToStr(); i.e. h.fit(t) is h returns True
    h.fit_transform(t)  # this returns the string casting of t
    
    """
    def fit(self, x, y=None):        
        return self
        
    def transform(self, x):        
        return x.astype(str)


u = MapToStr()
v = LabelBinarizer()
x = u.fit_transform(t)
y = v.fit_transform(x)
for i, a in enumerate(t):
    print(a, ' : ', y[i,:])

print('testing transform...')
for s in ss:
    print('.')
    z = v.transform(u.transform(s))
    for i, a in enumerate(s):
        print(a, ' : ', z[i,:])

    
    
# Method 4, cf. https://github.com/scikit-learn/scikit-learn/issues/3956
#==============================================================================
# print('Method 4')
# from sklearn.pipeline import Pipeline
# p = Pipeline([('a', LabelEncoder()),('b', OneHotEncoder(sparse=False))])
# p.fit_transform()
#==============================================================================

# Method 9
print('Method 9: Use pandas')

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
class ToDummiesWrapper(BaseEstimator, TransformerMixin):
    """
    cf. http://stackoverflow.com/questions/21057621/sklearn-labelencoder-with-never-seen-before-values
    """
    def fit(self, X):            
        self.columns = pd.get_dummies(X, dummy_na=True).columns     
        return self
        
    def transform(self, X):  
        # print('in transform')
        # print(pd.get_dummies(X))
        v = pd.get_dummies(X, dummy_na=True).reindex(columns=self.columns, fill_value=0.)
        return v.as_matrix()



v = ToDummiesWrapper()
y = v.fit_transform(t)
for i, a in enumerate(t):
    print(a, ' : ', y[i,:])

print('testing transform...')
for s in ss:
    print('.')
    z = v.transform(s)
    for i, a in enumerate(s):
        print(a, ' : ', z[i,:])
        
        