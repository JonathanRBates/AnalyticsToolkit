# -*- coding: utf-8 -*-
"""
Join individual Project tables into single dataframe.
Save table summaries in ./data_profiles/

Created on Tue Jan 10 13:54:20 2017

@author: jb2428
"""

from __future__ import print_function
import pandas as pd
import numpy as np
import os

import pickle as pk
# import cPickle as pk

PK_DATA_FILE = 'main.data'


##############################################################################
# Outcomes
##############################################################################

def gen_outcomes(df_main):
    
    if df_main is None:
        df_main = pk.load(open(PK_DATA_FILE, 'rb'))
    
    # 10-year risk
    nyear = 10.
    df_main['OUTCOME'] = ((df_main['A'] == 1) & (df_main['ADY'] < 365.25*nyear)) | \
        ((df_main['B'] == 1) & (df_main['BDY'] < 365.25*nyear))
          
    return df_main


##############################################################################


def init_df_main(LOAD_FLAG = False):
    """
    
    
    """
    
    if os.path.isfile(PK_DATA_FILE) and LOAD_FLAG:
    
        print("Loading from Pickle file..")
        df_main = pk.load(open(PK_DATA_FILE, 'rb'))
        
    else:
    
        # Variable, File definitions
        v_pth = 
        v_file = 'Codebook.xlsx'    
        
        print('...')
                
        # The order of files is important if doing left-joins...
        # Putting outcomes at end of list...
        files = ['data1.csv', 
                 'data2.csv',
                 'data3.csv'
                 ]
            
        # convert to dataframes        
        dfs = [pd.read_csv(os_path+'data1.csv')]
        dfs.append(pd.read_csv(os_path+'data2.csv'))
        dfs.append(pd.read_csv(os_path+'data3.csv'))       
        
        prefixes = [file.split('_')[0] for file in files]                           
                    
        # rename variables ################################### 
        for i, pre in enumerate(prefixes):
            ll = dict((b, pre+'_'+b) for b in dfs[i].columns.tolist())
            ll['ID'] = 'ID'
            dfs[i] = dfs[i].rename(columns=ll)
        
        # write file summaries to excel sheet ################################### 
        for i, file in enumerate(files):
            fname = './data_profiles/'+prefixes[i]+'.xlsx'
            generate_file_summary(dfs[i], fname)                      
        
        # join data frames #######################################################
        print('...joining all dataframes...')
        df_main = dfs[0]
        for i in range(1,len(dfs)):            
            df_main = pd.merge(df_main, dfs[i], left_on='ID', right_on='ID', how='left')
            print('.',sep='')
        
        # write joined data frame summary to excel sheet ########################
        generate_file_summary(df_main, './data_profiles/main.xlsx')
        
        # pickle joined data frame ###############################################
        df_main = df_main.set_index('ID')
        
        # Look at duplicates
        # print(df[df.index.duplicated(keep=False)].head(n=20))     
      
    return df_main
    
    
##############################################################################
# Add new data
##############################################################################
def append_table(df_main, file):
    # Variable, File definitions
    df = pd.read_csv(os_path+file)   
    prefix = file.split('.')[0]
    df_main = pd.merge(df_main, df, left_index=True, right_on='ID', how='left', suffixes=('_main','_'+prefix))        
    return df_main

##############################################################################
# Helper Functions
##############################################################################


def generate_file_summary(df,fname):
    """
    :df: dataframe to summarize
    :fname: filename to save as
    """
    writer = pd.ExcelWriter(fname, engine='xlsxwriter')              
    df.head().to_excel(writer, sheet_name='head')
    # TODO: if index is not ID...
    pd.DataFrame(df['ID'].value_counts()).head(n=10).to_excel(writer, sheet_name='ID counts')    
    df_stats = df.apply(lambda x: pd.Series((pd.isnull(x).sum(), 
                                             np.max(x) if (x.dtype != object) else None, 
                                             np.min(x) if (x.dtype != object) else None,
                                             np.mean(x) if (x.dtype != object) else None),
                                            index=['missing count',
                                            'max',
                                            'min',
                                            'mean']
                                            )).transpose()
    # df_stats = df.describe().transpose()
    df_stats.to_excel(writer, sheet_name='column stats')
    pd.DataFrame(list(df.shape),index=('dim 0','dim 1'),columns=['']).to_excel(writer, sheet_name='table shape')
    writer.save()
    return None
    

    
##############################################################################


if __name__ == "__main__":
    
    print('Initializing df_main...')
    df_main = init_df_main(LOAD_FLAG = False)
    print('Saving initial df_main...')
    pk.dump(df_main, open(PK_DATA_FILE, 'wb'))  
    print('Adding outcomes to df_main...')
    df_main = gen_outcomes(df_main) 
    print('Saving df_main with outcomes...')
    pk.dump(df_main, open(PK_DATA_FILE, 'wb'))  
    generate_file_summary(df_main.reset_index(),'./data_profiles/main_full.xlsx')  
    pk.dump(df_main.head(n=500), open('main_cut.data', 'wb'))      
    
    #  append_table(df_main, file)
