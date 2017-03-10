# -*- coding: utf-8 -*-
"""
Test DataFrameMapper/Pipeline preprocessing for heterogeneous data.

The following preprocessing tools all seem to have some problems 
with the tests I've done:
    LabelBinarizer
    OneHotEncoder
    LabelEncoder
    MultilabelEncoder

See the test_categorical_mappers*.py scripts.

Created on Sun Jan 15 16:06:52 2017

@author: jb2428
"""

import pandas as pd
import numpy as np
np.set_printoptions(precision=4, linewidth=180)

df_train = pd.DataFrame({
             'A':[np.nan,0.,2.,0.,2.,0.,2.],
             'B':[np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan], 
             'C':[0,5,5,5,5,0,5],                 
             'D':[0,1.,0,np.nan,0,1.,0],
             'E':[np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan],             
             'F':['a','a',1,1,1,'a',1],
             'G':[np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan]
           })
           
           
df_test = pd.DataFrame({
             'A':[np.nan,0.,0.,0.,0.,0.,0.],             
             'B':[np.nan,4,np.nan,np.nan,np.nan,np.nan,np.nan], 
             'C':[0,np.nan,5,0,5,0,0],
             'D':[0,0,np.nan,0,0,0,0],
             'E':[np.nan,4,np.nan,np.nan,np.nan,np.nan,np.nan],
             'F':['a','a','a',np.nan,'a','a','novel'],
             'G':[np.nan,4,np.nan,np.nan,np.nan,np.nan,np.nan]
           })
   
##############################################################################

from sklearn.base import BaseEstimator, TransformerMixin

import pandas as pd

class ToDummiesWrapper(BaseEstimator, TransformerMixin):
    """
    cf. http://stackoverflow.com/questions/21057621/sklearn-labelencoder-with-never-seen-before-values
    """
    def fit(self, X):         
        # catch all nan case
        # e.g. pd.get_dummies(pd.DataFrame(5*[np.nan]),dummy_na=True)
        # does not act as expected...
        u = pd.DataFrame(X)
        self.AllNAN_flag = pd.isnull(u).all().all()        
        if self.AllNAN_flag:            
            self.classes_ = ['NaN']
        else:
            self.classes_ = pd.get_dummies(u, dummy_na=True, prefix='', prefix_sep='').columns           
        return self
        
    def transform(self, X):       
        if self.AllNAN_flag:            
            return np.zeros(shape=X.shape)
        else:            
            v = (
                 pd.get_dummies(pd.DataFrame(X), 
                                dummy_na=True, 
                                prefix='', 
                                prefix_sep='')
                                .reindex(columns=self.classes_, 
                                         fill_value=0.)                 
                )        
            return v.as_matrix()

 
class CatchAllNAN(BaseEstimator, TransformerMixin):
    """Replace w zeros in Pipeline/DataFrameMapper if all nan.
    
    This is created as a workaround to errors in DataFrameMapper() /Pipeline()
    when an entire column is missing and the Imputer() throws an error.    
    Note that if all nan is encountered in fitting, but there are non-nan
    values in transform input, the transform will convert these to 0s.
    """
    def fit(self, X):      
        self.AllNAN_flag = pd.isnull(pd.DataFrame(X)).all().all()
        return self
        
    def transform(self, X):          
        if self.AllNAN_flag:                
            return  np.zeros(shape=X.shape)
        else:
            return X        
            
            
pd.get_dummies(pd.DataFrame(5*[np.nan]),dummy_na=True)

            
            
def preprocess_train_data(x, tr):
    """
    Impute, binarize, scale an input dataframe. Save the transformation.
    
    :param pandas.DataFrame x: dataframe to preprocess
    :param tuple tr: the transformation rule/code for preprocessing
    :return: the preprocessed dataframe xt and transformation details
    """
        
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, Imputer
    from sklearn_pandas import DataFrameMapper
    
    map_instructions = list()
    
    if 'continuous_vars' in tr:
        map_instructions.extend([([v], [CatchAllNAN(), Imputer(strategy='mean'), StandardScaler()]) for v in tr['continuous_vars']])
    
    if 'categorical_vars' in tr:
        # map_instructions.extend([([v], [MapToStr(), LabelBinarizer()]) for v in tr['categorical_vars']])
        map_instructions.extend([([v], ToDummiesWrapper()) for v in tr['categorical_vars']])        
        
    if 'binary_vars' in tr:
        map_instructions.extend([([v], [CatchAllNAN(), Imputer(strategy='most_frequent'), MinMaxScaler()]) for v in tr['binary_vars']])
    
    mapper = DataFrameMapper(map_instructions)

    xt = mapper.fit_transform(x)
        
    # get column names
    column_names = list()
    for feature in mapper.features: 
        has_classes_flag = getattr(feature[1], "classes_", None)
        original_feature_name = feature[0][0]        
        if has_classes_flag is None:
            column_names.extend(original_feature_name)          
        else:              
            class_names = feature[1].classes_
            column_names.extend([original_feature_name+'_'+str(sub) for sub in class_names])       
        
    xt = pd.DataFrame(xt, columns=column_names)
    return xt, mapper, column_names
    
def preprocess_test_data(x, mapper, column_names):
    """
    Apply transformation learned on training set with preprocess_train_data()
    
    :param pandas.DataFrame x: test dataframe to preprocess
    :param . mapper: 
    :param . column_names:
    :return: the preprocessed dataframe 
    """
    return pd.DataFrame(mapper.transform(x), columns=column_names)
    
##############################################################################

transformation_rules = {'continuous_vars': ['A', 'B'], 
                        'binary_vars': ['C','D','E'],
                        'categorical_vars': ['F','G'],
                        }


print('Training', '.'*60)                        
print(df_train)     
print('.'*100)     
xt, mapper, column_names = preprocess_train_data(df_train, transformation_rules)
print(xt)
print('.'*100)  
print('Testing', '.'*60)   
print(df_test)
print('.'*100)  
xv = preprocess_test_data(df_test, mapper, column_names)
print(xv)
print('.'*100)   
