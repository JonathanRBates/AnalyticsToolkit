# -*- coding: utf-8 -*-
"""
2017-01-15

Predictive Analytics Toolkit


"""

from __future__ import print_function

import pandas as pd
import numpy as np
import pickle as pk


##############################################################################
# Read Excel Worksheet
##############################################################################

def get_transformation_rules(f_codebook):   
    r"""
    Read the feature transformation rules from the excel file specified by f_codebook.    
    
    :return: the transformation rule/code for preprocessing
    
    
    Example.
    
    f_codebook = r'location\file_name.xlsx'
    transformation_rules = get_transformation_rules(f_codebook)
    print(transformation_rules)   
    
    """
    df = pd.read_excel(f_codebook)
     
    inc = df['Predictor']==1.
    cat = df['Mapping (u=continuous, b=binary, g=categorical)']  
    
    def pname(r):
        pre = r['File'].split('.')[0]
        b = r['Variable Name']
        # return pre+'_'+b    # <- use this when Variable Name is not unique
        return b
        
    # append prefix
    include_cols = df[inc].apply(pname, axis=1).tolist()           
    categorical_vars = df[inc &(cat=='g')].apply(pname, axis=1).tolist()    
    binary_vars = df[inc &(cat=='b')].apply(pname, axis=1).tolist()   
    continuous_vars = df[inc &(cat=='u')].apply(pname, axis=1).tolist()   
    
    transformation_rules = {'continuous_vars': continuous_vars, 
        'categorical_vars': categorical_vars,
        'binary_vars': binary_vars}
       
    return include_cols, transformation_rules


##############################################################################
# Data Preprocessing for Train/Test Data
##############################################################################

def preprocess_train_data(x, tr=None):
    r"""
    Impute, binarize, scale an input dataframe. Save the transformation.
    
    :param pandas.DataFrame x: dataframe to preprocess
    :param tuple tr: the transformation rule/code for preprocessing
    :return: the preprocessed dataframe xt and transformation details
    
    
    Example.

    df_train = pd.DataFrame({
             'A':[np.nan,100.,100.,100.,100.,100.,100.],
             'B':[103.02,107.26,110.35,114.23,114.68,97.,np.nan], 
             'C':['big','small','big','small','small','medium','small'],
             'D':[0,5,5,5,5,0,5],
             'E':[0,1,0,np.nan,0,1,0],
             'F':[0,1,1,1,1,0,1]
           })
           
           
    df_test = pd.DataFrame({
                 'A':[100.,110.,110.,np.nan,110.,110.,110.],
                 'B':[.02,.26,.35,.23,.68,97.,np.nan], 
                 'C':['small','small','big','small','giant','medium','small'],
                 'D':[0,np.nan,5,0,5,0,0],
                 'E':[0,0,np.nan,0,0,0,0],
                 'F':[0,0,0,0,0,0,0]
               })    
    
    transformation_rules = {'continuous_vars': ['A', 'B'], 
        'categorical_vars': ['C', 'F'],
        'binary_vars': ['D','E']}
    
    xt, mapper, column_names = preprocess_train_data(df_train, transformation_rules)    
    xv = preprocess_test_data(df_test, mapper, column_names)    
    
    
    """
    
    # If no transformation rules, return the original data
    if tr is None:
        return x, None, list(x.columns.values)
    
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, Imputer
    from sklearn_pandas import DataFrameMapper
    
    # TODO: consider using feature union for this...
    # TODO: should all nans be thrown out at an earlier stage of analysis?
    #
    
    map_instructions = list()
    
    if 'continuous_vars' in tr:
        map_instructions.extend([([v], [CatchAllNAN(), Imputer(strategy='mean'), StandardScaler()]) for v in tr['continuous_vars'] if v in x.columns])
    
    if 'categorical_vars' in tr:
        # map_instructions.extend([([v], [MapToStr(), LabelBinarizer()]) for v in tr['categorical_vars']])
        # TODO: REMOVED dummy_na coding to help elimate colinearity, but there must be a better way to do this...
        map_instructions.extend([([v], ToDummiesWrapper(dummy_na=True)) for v in tr['categorical_vars'] if v in x.columns])        
        
    if 'binary_vars' in tr:
        map_instructions.extend([([v], [CatchAllNAN(), Imputer(strategy='most_frequent'), MinMaxScaler()]) for v in tr['binary_vars'] if v in x.columns])
    
    mapper = DataFrameMapper(map_instructions)
    xt = mapper.fit_transform(x)      
   
    # get column names
    column_names = list()
    for feature in mapper.features: 
        has_classes_flag = getattr(feature[1], "classes_", None)
        original_feature_name = feature[0][0]        
        if has_classes_flag is None:
            # print(original_feature_name)
            column_names.append(original_feature_name)          
        else:              
            class_names = feature[1].classes_
            column_names.extend([original_feature_name+'_'+str(sub) for sub in class_names])       
      
    # xt['ID'] = x.index
    # xt = pd.DataFrame(xt, columns=column_names, index='ID')
    return xt, mapper, column_names
    
    
    
def preprocess_test_data(x, mapper, column_names):
    r"""
    Apply transformation learned on training set with preprocess_train_data()
    
    :param pd.core.frame.DataFrame x: test dataframe to preprocess
    :param ? mapper: 
    :param list column_names:
    :return: the preprocessed dataframe 
    
    """
    
    # If mapper is None, return the original data
    if mapper is None:
        return x
    
    return pd.DataFrame(mapper.transform(x), columns=column_names) 
    
    
    
def get_interactions(X, column_names, degree=2):   
    """
    With scaling, should include per CV fold
    
    Include in preprocessing, so variables to interact can be specified in transformation rules.
    """
    # TODO: how does this fit into pipeline?
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.preprocessing import scale
    poly = PolynomialFeatures(degree=degree, interaction_only=True, include_bias=False)
    X = poly.fit_transform(X)    
    scale(X, axis=0, with_mean=False, with_std=False, copy=False) 
    column_names = ['*'.join(np.array(column_names)[b==1]) for b in poly.powers_]   
    
    
##############################################################################
# Cross Validation
##############################################################################

# TODO: replace with scikit learn's CV module...
def metamodels_cross_validate(X, y, transformation_rules=None, metamodels=None, kfolds=10, f_validate=None, verbose=False):
    """
    Run k-fold cross validation for metamodels; binary classifiers.
    
    :param pd.core.frame.DataFrame X: the training data n, p
    :param pd.core.series.Series y: the training labels n,
    :param <list of metamodels> metamodels: [see below]
    :param int kfolds: number of folds for validation
    :param str f_validate: the file name to pickle results
    
    """
        
    assert isinstance(X, pd.core.frame.DataFrame), "Training data must be a pandas DataFrame"
    assert isinstance(y, pd.core.series.Series), "Training labels must be a pandas Series"

    from sklearn.metrics import roc_auc_score
    from collections import defaultdict
    
    # make CV splits evenly wrt proportion of outcomes
    # TODO: add alternative CV method with boostrap resampling
    #?? sys.version_info.major == 3
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=kfolds, random_state=1)
    cv_splits = list(skf.split(X, y))
        
    all_results = defaultdict(tuple)
    for m in metamodels: 
        print('Training', m['id'], '-'*20)
        results_ = defaultdict(list)
        htable_ = indexed_dict_product(m['hyperparameters']) # keep outside of CV
        print('Fold', end='  ')
        for k, (train_fold_index, test_fold_index) in enumerate(cv_splits):
            print(k, end='')                        
            Xt = X.iloc[train_fold_index,:]
            Xf = X.iloc[test_fold_index,:]
            yt = y.iloc[train_fold_index]
            yf = y.iloc[test_fold_index]         
            # print('Train fold count:\n', yt.value_counts())
            # print('Test fold count:\n', yf.value_counts())
            # Standardize Data
            Xt, mapper, column_names = preprocess_train_data(Xt, transformation_rules)    
            Xf = preprocess_test_data(Xf, mapper, column_names)
            for i in htable_:
                print('.',end='')
                # TODO: include options for when .fit is not method used
                clf = m['model'](**htable_[i])
                # Fit Model 
                clf = clf.fit(Xt, yt)   
                # Assess Model
                # TODO: have optimality criterion per metamodel
                auc = roc_auc_score(yf, clf.predict_proba(Xf)[:,1])
                results_[i].append({'fold':k,'auc':auc}) 
            #/Hyperparameter Search
        print() 
        #/KFold Crossvalidation   
        all_results[m['id']] = (htable_, results_)
        idx_best, hyp_best, results_best = summarize_cv_results(all_results[m['id']], verbose)
        print('Best:',hyp_best)
        print('     ',results_best)
        # Save current results
        f_validate_handle = open(f_validate, 'wb')
        pk.dump((all_results,metamodels,cv_splits,transformation_rules), f_validate_handle)  
        f_validate_handle.close()       
        
    return None          
  

def summarize_cv_results(results, verbose=False):
    """
    results = all_results[m['id']]
    """       
    results_summary = dict((i,{
                                'auc_mean': np.mean([x['auc'] for x in kaucs]),
                                'auc_std': np.std([x['auc'] for x in kaucs]),
                                'auc_percentile_50': percentile(50)([x['auc'] for x in kaucs])                 
                              }) for i, kaucs in results[1].items())   
    results_summary = pd.DataFrame(results_summary).transpose()    
    idx_best = results_summary['auc_mean'].idxmax()   # a not necessarily unique best classifier
    if verbose:
        print(results_summary.sort_values(by='auc_mean'))                    
    # print(results[0][idx_best])
    # print(results_summary.loc[idx_best])
    return idx_best, results[0][idx_best], results_summary.loc[idx_best]


##############################################################################
# Final Training
##############################################################################

def fit_optimal_model_to_training_data(X_train, y_train, X_test, y_test, f_validate=None, f_fit_models=None):
    """
    e.g.
    'metamodels_cross_validate_results.data'
    'fit_models.data'
    """
    from sklearn.metrics import roc_auc_score
    with open(f_validate, 'rb') as f:                                                                 
        all_results, metamodels, cv_splits, transformation_rules = pk.load(f)  
    Xt, mapper, column_names = preprocess_train_data(X_train, transformation_rules)    
    Xf = preprocess_test_data(X_test, mapper, column_names)
    fit_models = dict()    
    for m in metamodels: 
        print('Testing', m['id'], '-'*20)
        _, hyp_best, _ = summarize_cv_results(all_results[m['id']])
        print('Hyperparameters:')        
        for key, value in hyp_best.items() :
            print('    {} : {}'.format(key, value))
        clf = m['model'](**hyp_best)
        # Fit Model 
        clf = clf.fit(Xt, y_train) 
        fit_models[m['id']] = clf
        # Save Model        
        f_fit_models_handle = open(f_fit_models, 'wb')
        pk.dump((mapper, column_names, fit_models), f_fit_models_handle)
        f_fit_models_handle.close()     
        # Assess Model
        auc = roc_auc_score(y_test, clf.predict_proba(Xf)[:,1])
        print('>> AUC: {:5.3f}'.format(auc))
    return None        

##############################################################################
# Testing
##############################################################################   
     
def summarize_test_results(X_test, y_test, f_validate=None, f_fit_models=None):
    """
    e.g.
    'metamodels_cross_validate_results.data'
    'fit_models.data'
    """
    from sklearn.metrics import roc_auc_score
    with open(f_validate, 'rb') as f:                                                                 
        all_results, metamodels, cv_splits, transformation_rules = pk.load(f)  
    with open(f_fit_models, 'rb') as f:
        mapper, column_names, fit_models = pk.load(f) 
    Xf = preprocess_test_data(X_test, mapper, column_names)
    for m in metamodels: 
        print('Testing', m['id'], '-'*20)
        _, hyp_best, _ = summarize_cv_results(all_results[m['id']])
        print('Hyperparameters:')        
        for key, value in hyp_best.items() :
            print('    {} : {}'.format(key, value))
        clf = fit_models[m['id']]
        # Assess Model
        auc = roc_auc_score(y_test, clf.predict_proba(Xf)[:,1])
        print('>> AUC: {:5.3f}'.format(auc))
    return None       
    
##############################################################################
# Get Calibration
##############################################################################

def get_calibration():
    r"""
    
    """
    # TODO:
    # y = true outcome
    # r = risk score
    # reindex y, r by r ...    
    return None

##############################################################################
# Data Simulation for Unit Testing, Operating Characteristics Testing
##############################################################################

def simulate_two_class(cluster_size = [100,20]):
    r"""
    Generate separable, imbalanced data with two classes in two dimensions.    
    
    cluster_size is the number of samples per class.
    """    
    
    mean1 = [0,0]
    mean2 = [2,2]
    cov1 = [[0.5,0],[0,0.5]]
    cov2 = [[1,0],[0,1]]   
    
    X = np.vstack([
            np.random.multivariate_normal(mean1, cov1, cluster_size[0]),
            np.random.multivariate_normal(mean2, cov2, cluster_size[1])
        ])
        
    y = np.vstack([
            np.ones((cluster_size[0],1)),
            -1*np.ones((cluster_size[1],1))
        ]).ravel()
        
    X = pd.DataFrame(X, columns=['A','B'])
    y = pd.DataFrame(y, columns=['O'], dtype=np.int32)
           
    # GET EQUAL SPLITS OF CASES IN TRAIN/TEST
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.6,stratify=y)     
      
    print('Train count:\n', y_train['O'].value_counts())
    print('Test count:\n', y_test['O'].value_counts())
        
    import matplotlib.pyplot as plt

    colormap = {1:'g', -1:'r'}        
    
    fig = plt.figure()
    fig.add_subplot(121)
    for (x1,x2,yval) in zip(X_train['A'], X_train['B'], y_train['O']):
        plt.plot(x1, x2, 'o', color=colormap[yval])  
    plt.title('Training Data')
    plt.axis('equal')
    fig.add_subplot(122)
    for (x1,x2,yval) in zip(X_test['A'], X_test['B'], y_test['O']):
        plt.plot(x1, x2, 'o', color=colormap[yval])  
    plt.title('Testing Data')
    plt.axis('equal')
    plt.show()  
    
    return X_train, X_test, y_train, y_test


##############################################################################
# Custom Preprocessing Classes
##############################################################################
    
from sklearn.base import BaseEstimator, TransformerMixin

class ToDummiesWrapper(BaseEstimator, TransformerMixin):
    """
    cf. http://stackoverflow.com/questions/21057621/sklearn-labelencoder-with-never-seen-before-values
    
    1/27/2017 - modified to reduce colinearity by appending [1:] to columns
    
    """
    def __init__(self, dummy_na=False):
        self.dummy_na = dummy_na  
        
    def fit(self, X):     
        if isinstance(X, pd.core.series.Series):
            self.classes_ = pd.get_dummies(X, dummy_na=self.dummy_na, prefix='', prefix_sep='').columns[1:] 
            self.X_type = pd.core.series.Series
        elif isinstance(X, np.ndarray) and X.shape[1]==1:            
            self.classes_ = pd.get_dummies(pd.Series(X.ravel()), dummy_na=self.dummy_na, prefix='', prefix_sep='').columns[1:]
            self.X_type = np.ndarray
        else:
            assert "Check type(X)", type(X)                      
        return self
        
    def transform(self, X):        
        if self.X_type == pd.core.series.Series:            
            v = (
                 pd.get_dummies(X, 
                                dummy_na=self.dummy_na, 
                                prefix='', 
                                prefix_sep='')
                                .reindex(columns=self.classes_, 
                                         fill_value=0.)                 
                )        
        elif self.X_type == np.ndarray and X.shape[1]==1: 
            v = (
                 pd.get_dummies(pd.Series(X.ravel()), 
                                dummy_na=self.dummy_na, 
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
    # TODO: do we really want to use this?
    # Should all nans be thrown out at an earlier stage of analysis?
    #
    def fit(self, X):      
        self.AllNAN_flag = pd.isnull(pd.DataFrame(X)).all().all()
        return self
        
    def transform(self, X):          
        if self.AllNAN_flag:                
            return  np.zeros(shape=X.shape)
        else:
            return X           
    
##############################################################################
# Basic Helper Functions
##############################################################################

def percentile(n):
    def percentile_(x):
        return np.percentile(x, n)
    percentile_.__name__ = 'percentile_%s' % n
    return percentile_
    

def indexed_dict_product(d):   
    """
    Return a dictionary for the cartesian product of values for dictionary d. 
    
    Note. Used in iterating through hyperparameters.
    
    """    
    def dict_product(d):
        """
        Return the cartesian product of values for dictionary d.    
        """
        import itertools
        return (dict(zip(d, x)) for x in itertools.product(*d.values()))
    return dict((i, x) for i, x in enumerate(dict_product(d)))

      
