# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 14:11:04 2017

@author: jb2428
"""

from __future__ import print_function

import pandas as pd
import numpy as np

import pickle as pk


##########################

def summary_missing(x):
    return pd.Series(pd.isnull(x).sum()/len(x), index=['missing fract'])

def summary_continuous(x):
    return pd.Series((pd.isnull(x).sum()/len(x), 
                     np.sum(x) if (x.dtype != object) else None,
                     np.max(x) if (x.dtype != object) else None, 
                     np.min(x) if (x.dtype != object) else None,
                     np.mean(x) if (x.dtype != object) else None),
                     index=['missing fract',
                         'sum',                                               
                         'max',
                         'min',
                         'mean'])

##########################




DATA = 'train_test_split.data'
X_train, X_test, y_train, y_test, tr = pk.load(open(DATA, 'rb'), encoding='latin1')

print('Train count:\n', y_train.value_counts())
print('Test count:\n', y_test.value_counts())

# from aggregate_tables import generate_file_summary
# generate_file_summary(X_train.reset_index(),'train_set_initial.xlsx')
writer = pd.ExcelWriter('train_set.xlsx', engine='xlsxwriter')    

df_stats1 = X_train.apply(summary_missing).transpose()

df_stats2 = X_test.apply(summary_missing).transpose()

df_stats = pd.merge(df_stats1, df_stats2, how='left', left_index=True, right_index=True, suffixes=(' (train)',' (test)'))

# Add descriptions
df = pd.read_excel('codebook.xlsx')
def pname(r):
    pre = r['Project File'].split('_')[0]
    b = r['Variable Name']
    return pre+'_'+b

# append prefix
df['pkey'] = df.apply(pname, axis=1).tolist()
df = df.set_index(['pkey'])[['Description']]
df_stats = pd.merge(df_stats, df, how='left', left_index=True, right_index=True)
df_stats.to_excel(writer, sheet_name='column stats (raw)')
        

from analytic_toolkit import preprocess_train_data, preprocess_test_data
Xt, mapper, column_names = preprocess_train_data(X_train, tr)  
Xt = pd.DataFrame(Xt, columns=column_names, index=X_train.index)
Xf = preprocess_test_data(X_test, mapper, column_names)
Xf = pd.DataFrame(Xf, columns=column_names, index=X_train.index)

df_stats1 = Xt.apply(summary_continuous).transpose()

df_stats2 = Xf.apply(summary_continuous).transpose()

df_stats = pd.merge(df_stats1, df_stats2, how='left', left_index=True, right_index=True, suffixes=(' (train)',' (test)'))
df_stats.to_excel(writer, sheet_name='column stats (processed)')


# X_train.head(n=10).to_excel(writer, sheet_name='head (raw)')          
# Xt.head(n=10).to_excel(writer, sheet_name='head (processed)')          


writer.save()

