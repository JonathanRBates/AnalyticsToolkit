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


t = np.array(['a','a','A','A','a'])   # Methods work, but difference in output number of dimensions
# t = np.array([0,3,0,3,0,0,2,2,2])   # Methods are equivalent
# t = np.array([0,3,0,3,0,0,np.nan,0,np.nan])   # Method 3 is optimal
# t = np.array(['a',np.nan,np.nan,'a',34.,34.])  # Methods are equivalent
# t = np.array([0.,3.,0.,3.,0.,0.,np.nan,0.,np.nan])   # Method 3, 7 is optimal


# Method 1  -- THIS METHOD CAN'T HANDLE NOVEL DATA IN TRANSFORM
print('Method 1: LabelEncoder+OneHotEncoder')
u = LabelEncoder()
v = OneHotEncoder(sparse=False)
w = u.fit_transform(t)
g = v.fit_transform(w.reshape(-1,1))
for i, a in enumerate(t):
    print(a, ' : ', g[i,:])


# Method 2
print('Method 2: LabelBinarizer without type casting')
y = LabelBinarizer()
try:
    g = y.fit_transform(t)
    for i, a in enumerate(t):
        print(a, ' : ', g[i,:])
except:
    print('skipping...\n')       
   
    
# Method 3
print('Method 3: LabelBinarizer with type casting')
g = y.fit_transform(t.astype(str))
for i, a in enumerate(t):
    print(a, ' : ', g[i,:])
    
    
# Method 4, cf. https://github.com/scikit-learn/scikit-learn/issues/3956
#==============================================================================
# print('Method 4')
# from sklearn.pipeline import Pipeline
# p = Pipeline([('a', LabelEncoder()),('b', OneHotEncoder(sparse=False))])
# p.fit_transform()
#==============================================================================
    
# Method 5
print('Method 5: CountVectorizer')
#==============================================================================
# from sklearn.feature_extraction.text import CountVectorizer
# q = CountVectorizer()
# # g = q.fit_transform(t).todense()
# g = q.fit_transform(t.astype(str)).todense()
# for i, a in enumerate(t):
#     print(a, ' : ', g[i,:])
#==============================================================================
 
# Method 6
print('Method 6: MultiLabelBinarizer -- how is this coding happening?')
#==============================================================================
# from sklearn.preprocessing import MultiLabelBinarizer
# mlb = MultiLabelBinarizer()
# g = mlb.fit_transform(t.astype(str))
# for i, a in enumerate(t):
#     print(a, ' : ', g[i,:]) 
#     
#==============================================================================
    
    
# Test Method 3 as a Pipeline transformation

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
      
       
#==============================================================================
# import pandas as pd
# class MapToStr(BaseEstimator, TransformerMixin):
#     """
#     
#     """
#     def fit(self, x, y=None):        
#         return self
#         
#     def transform(self, x):        
#         return x.astype(str)       
#==============================================================================

from sklearn.preprocessing import LabelBinarizer     
class LabelBinarizerExp(LabelBinarizer):
    """
    http://stackoverflow.com/questions/31947140/sklearn-labelbinarizer-returns-vector-when-there-are-2-classes
    """
    def transform(self, y):
        Y = super().transform(y)
        if self.y_type_ == 'binary':
            return np.hstack((Y, 1-Y))
        else:
            return Y

    def inverse_transform(self, Y, threshold=None):
        if self.y_type_ == 'binary':
            return super().inverse_transform(Y[:, 0], threshold)
        else:
            return super().inverse_transform(Y, threshold)
            
            
            
# Method 3
print('Method 7: LabelBinarizerExp with type casting')
x = LabelBinarizerExp()
g = x.fit_transform(t.astype(str))
for i, a in enumerate(t):
    print(a, ' : ', g[i,:])            