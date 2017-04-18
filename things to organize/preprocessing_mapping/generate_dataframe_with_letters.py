# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 13:15:34 2017

@author: jb2428
"""

import numpy as np
import pandas as pd
import string

N = 100

L = np.random.choice([a for a in string.ascii_uppercase[:3]],size=(N,2),replace=True)
L = pd.DataFrame(L,columns=['col1','col2'])
L['fun1'] = list(map(lambda x, y: x+'.'+y,L['col1'], L['col2']))

print(L.head(n=6))



L['col3'] = np.random.randint(0,3,size=(N,1))



L['fun2'] = list(map(lambda x, y: (x not in ['B', 'C']) & (y==1.).item(), L['col1'], L['col3']))
L['fun3'] = L.apply(lambda x: (x['col1'] not in ['B', 'C']) & (x['col3']==1.), axis=1)    
print(L.head(n=6))