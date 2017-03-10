# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 12:56:26 2017

@author: jb2428
"""

import numpy as np
import pandas as pd
from string import ascii_uppercase as UCL
from analytic_toolkit import preprocess_train_data

np.set_printoptions(precision=2, linewidth=150, threshold=1000)

##############################################################################
# MAKE SOME DATA ("TRAINING DATA")
##############################################################################

N = 10000
p = 2
U = pd.DataFrame(np.random.random((N, p)), columns=['U.'+c for c in UCL[:p]])  # continuous covariates
L = pd.DataFrame(np.random.choice([a for a in UCL[:2]],size=(N,1),replace=True), columns=['CAT'])
X = pd.concat([U, L],axis=1)
print(X)

##############################################################################
# PREPROCESS
##############################################################################

transformation_rules = {'continuous_vars': U.columns.tolist(), 
        'categorical_vars': L.columns.tolist()}
        
Xt, mapper, column_names = preprocess_train_data(X, transformation_rules)

##############################################################################
# ADD POLYNOMIAL FEATURES
##############################################################################

from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import scale

poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
Xtt = poly.fit_transform(Xt)    
scale(Xtt, axis=0, with_mean=False, with_std=False, copy=False) 
column_names_poly = ['*'.join(np.array(column_names)[b==1]) for b in poly.powers_]
    
##############################################################################
# SIMULATE OUTCOMES GIVEN SPECIFICATION
##############################################################################
    
coef = np.zeros(Xtt.shape[1]+1)    
coef[4] = 5
coef[6] = 3
coef[9] = -10
xaug = np.hstack([np.ones((N,1)),Xtt])
proj = np.dot(xaug, coef)
r = 1./(1.+np.exp(-proj))
y = np.random.binomial(n=1, p=r)    
    
##############################################################################
# FIT MODELS
##############################################################################

import sklearn.linear_model as lm      
import matplotlib.pyplot as plt

n_runs = 10
n_C_values = 20
log_coefs_ = np.zeros((n_runs,n_C_values,Xtt.shape[1]))
C_ = np.logspace(-4,2,n_C_values)
for i in range(n_runs):
    for j in range(n_C_values):
        tlr = lm.LogisticRegression(penalty='l1', C=C_[j], fit_intercept=False)  
        tlr = tlr.fit(Xtt, y) 
        # log_coefs_.append(tlr.coef_.ravel().copy())    
        log_coefs_[i,j,:] = tlr.coef_.ravel()
  
        
##############################################################################
# GET VARIABLE ORDER
##############################################################################

log_coefs = np.mean(log_coefs_, axis=0)
theta = np.where(np.abs(log_coefs) > 1)
di = dict()
for i, j in zip(theta[0], theta[1]):
    print(i,j)
    if column_names_poly[j] not in di:
        di[column_names_poly[j]] = i

print(di)        

import operator
sorted_di = sorted(di.items(), key=operator.itemgetter(1))
print(sorted_di)

##############################################################################
# PLOT COEFFICIENTS
##############################################################################

colors = plt.get_cmap('jet')(np.linspace(0,1.,Xtt.shape[1]))

# plot all
fig = plt.figure()
ax = fig.add_subplot(111)    
fig.set_size_inches(8,6)

for i, col in enumerate(column_names_poly):
    plt.fill_between(C_,np.min(log_coefs_[:,:,i],axis=0),np.max(log_coefs_[:,:,i],axis=0),lw=1,alpha=0.8,color=colors[i],label=col+' ('+str(coef[i+1])+')')
ax.set_xscale('log')
plt.xlabel('C',fontsize=12)
plt.ylabel('Coefficient',fontsize=12)
# plt.ylim([-2,0.5])
# plt.axis('tight')
ax.legend(bbox_to_anchor=(1.05, 1.0), loc=2, borderaxespad=0.)
#plt.title('Covariate balance; ICD {}'.format(device))
#plt.savefig('C:\\Users\\jon.cmsT1b-PC\\Desktop\\python\\run_delta\\figures\\covariate_sdms_Run{suffix}.png'.format(suffix=suffix), bbox_inches='tight')    
plt.show()

