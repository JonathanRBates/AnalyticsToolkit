# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 10:42:50 2017

@author: jb2428
"""

from __future__ import print_function

ON_SERVER = True

# import os
# os.chdir(r'/home/jb2428')
 
import pickle as pk
from sklearn.cross_validation import train_test_split  # v2.7  <- use model_selection module if possible ; ? from __future__ import?
from analytic_toolkit import get_transformation_rules


##############################################################################
# ////////////////////////////////////////////////////////////////////////////
##############################################################################

project_reference_standard_file = 'project_train_test_split.data'

##############################################################################
# LOAD AGGREGATED DATA
##############################################################################

df_project = pk.load(open(r'project.data', 'rb'))

##############################################################################
# RECODE CATEGORICAL VALUES
##############################################################################

#di_VarX = {1.: 'Cat 1',
#             2.: 'Cat 2',
#             3.: 'Cat 3'  
#             }
#df_project.replace({'VarX': di_VarX}, inplace=True)
    
##############################################################################
# FEATURE ENGINEERING
##############################################################################

v_file = r'codebook.xlsx'    
     
include_cols, tr = get_transformation_rules(v_file) 

print('\n'.join(include_cols[:10]))
print('\n'.join(df_project.columns.tolist()[:20]))

X = df_project.loc[:,include_cols]
    
##############################################################################
# TRAIN/TEST SPLIT
##############################################################################

y = df_project.loc[:,'OUTCOME']   

# # REMOVE FREQUENTLY MISSING
# X.dropna(thresh=len(X) * 0.6, inplace=True, axis=1)  # filter out columns with more than 40% missing

# Split Data 
train_size = 0.25 # proportion of training samples
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, stratify=y, random_state=1)      # GET EQUAL SPLITS OF CASES IN TRAIN/TEST   
pk.dump((X_train, X_test, y_train, y_test, tr), open(project_reference_standard_file, 'wb'))  
   
print('Train count:\n', y_train.value_counts())
print('Test count:\n', y_test.value_counts())

from aggregate_tables import generate_file_summary
generate_file_summary(X_train.reset_index(),'project_train_set.xlsx')
 
