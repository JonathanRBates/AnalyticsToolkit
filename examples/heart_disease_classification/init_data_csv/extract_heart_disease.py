# -*- coding: utf-8 -*-
"""

Get initial data from

https://archive.ics.uci.edu/ml/datasets/Heart+Disease


"""

import pandas as pd

df = pd.read_csv(r'C:\Users\jb2428\Desktop\python\AnalyticsToolkit\heart_disease\extract_data\processed.cleveland.data',
                 header=None,
                 names=['age', 
                        'sex', 
                        'cp', 
                        'trestbps', 
                        'chol', 
                        'fbs', 
                        'restecg', 
                        'thalach', 
                        'exang',
                        'oldpeak',
                        'slope', 
                        'ca', 
                        'thal', 
                        'OUTCOME'])
                 
         
df.to_csv('data.csv',index=False)