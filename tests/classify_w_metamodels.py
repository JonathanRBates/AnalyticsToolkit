# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 19:12:14 2017

@author: jb2428
"""

from __future__ import print_function

import sys
import pickle as pk
# import cPickle as pk
from analytic_toolkit import metamodels_cross_validate, fit_optimal_model_to_training_data, summarize_test_results

##############################################################################

print(sys.version)

##############################################################################
# LOAD DATA
# cf create_reference_standard
##############################################################################

DATA = 'train_test_split_revised.data'
X_train, X_test, y_train, y_test, tr = pk.load(open(DATA, 'rb'))

print('Train count:\n', y_train.value_counts())
print('Test count:\n', y_test.value_counts())

##############################################################################
# Import Models
##############################################################################

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
# FF NNs
# XGBoost
# Gradient Boosting Classifier
# 
             
metamodels = [
              {'id': 'LogisticRegression',
               'model': LogisticRegression,
               'hyperparameters': {                 
                     'penalty': ['l1', 'l2'],
                     'class_weight': [None, 'balanced'],
                     'C': [1e-2, 1e-1, 1, 1e1, 1e2],
                     'n_jobs': [-1] }
                     },
             {'id': 'SGDClassifier',
               'model': SGDClassifier,
               'hyperparameters': {     
                     'loss' : ['log'],
                     'penalty' : ['elasticnet'],                                   
                     'alpha' : [1e-2, 1e-1, 1, 1e1, 1e2],
                     'l1_ratio' : [0.25, 0.5, 0.75],
                     'class_weight': ['balanced']}
                     },
             {'id': 'Random Forest',
               'model': RandomForestClassifier,
               'hyperparameters': {  
                     'n_estimators' : [10, 100, 1000],
                     'criterion': ['gini', 'entropy'],
                     'class_weight': [None, 'balanced'],
                     'max_depth': [None, 5, 10] }
                     },
             {'id': 'DecisionTreeClassifier',
               'model': DecisionTreeClassifier,
               'hyperparameters': {                 
                     'criterion': ['gini', 'entropy'],
                     'class_weight': [None, 'balanced'],
                     'max_depth': [None, 1, 2] }
                     },            
    # Larger C values take longer... (infinitely longer?)                    
              {'id': 'SVM.linear',
               'model': SVC,
               'hyperparameters': {  
                     'kernel' : ['linear'],                     
                     'class_weight': [None, 'balanced'],
                     'C': [1e-2, 1e-1, 1],
                     'probability': [True],
                     'cache_size': [10000]}
                     },
              {'id': 'SVM.poly',
               'model': SVC,
               'hyperparameters': {  
                     'kernel' : ['poly'],                     
                     'class_weight': [None, 'balanced'],
                     'C': [1e-2, 1e-1, 1],
                     'probability': [True],
                     'degree': [2, 3],
                     'cache_size': [10000]}
                     },
              {'id': 'SVM.rbf',
               'model': SVC,
               'hyperparameters': {  
                     'kernel' : ['rbf'],                     
                     'class_weight': [None, 'balanced'],
                     'C': [1e-2, 1e-1, 1],
                     'probability': [True],
                     'gamma': [1e-1, 1, 1e1],
                     'cache_size': [10000]}
                     }
             ]               
             
##############################################################################

metamodels_cross_validate(X_train, 
                          y_train, 
                          transformation_rules=tr, 
                          metamodels=metamodels, 
                          kfolds=10, 
                          f_validate='metamodels_cross_validate_results.data',
                          verbose=True)


fit_optimal_model_to_training_data(X_train, 
                                   y_train, 
                                   X_test, 
                                   y_test, 
                                   f_validate='metamodels_cross_validate_results.data', 
                                   f_fit_models='fit_models.data')


summarize_test_results(X_test, 
                       y_test, 
                       f_validate='metamodels_cross_validate_results.data', 
                       f_fit_models='fit_models.data')

