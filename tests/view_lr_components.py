# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 09:15:07 2017

@author: jb2428
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle as pk

poly_features = False

if poly_features:
    column_names_poly, log_coefs_ = pk.load(open('analysis.data', 'rb'), encoding='latin1')
    column_names = column_names_poly
    C_ = np.logspace(-4,4,20)
else:
     column_names, C_, log_coefs_ = pk.load(open('lr_l1_coefs.data', 'rb'))

##############################################################################
# NUMBER VARIABLES
##############################################################################

log_coefs = np.mean(log_coefs_, axis=0)
u = []
u.append(np.sum(np.abs(log_coefs),axis=1))
u.append(np.median(np.abs(log_coefs),axis=1))
u.append(np.sum(np.abs(log_coefs)>0.25,axis=1))
fig = plt.figure()
fig.set_size_inches(8,12)
for i, v in enumerate(u):
    ax = fig.add_subplot(str(311+i))
    plt.plot(C_,v,lw=1,alpha=0.8,label=i)
    ax.set_xscale('log')
    plt.xlabel('C',fontsize=12)
    plt.ylabel(i,fontsize=12)
# plt.ylim([-2,0.5])
# plt.axis('tight')
ax.legend(bbox_to_anchor=(1.05, 1.0), loc=2, borderaxespad=0.)
plt.show()

##############################################################################
# GET VARIABLE ORDER
##############################################################################

log_coefs = np.mean(log_coefs_, axis=0)
theta = np.where(np.abs(log_coefs) > 1.e-3)
di = dict()
max_num_vars = 40
iter_num_vars = 0
for i, j in zip(theta[0], theta[1]):
    # print(i,j)  # j indexes to predictor; i indexes C_ value that predictor appears
    if j not in di:
        iter_num_vars =iter_num_vars+1
        di[j] = i
        if iter_num_vars > max_num_vars:
            break

##############################################################################
# PLOT COEFFICIENTS
##############################################################################

colors = plt.get_cmap('jet')(np.linspace(0,1.,len(di)))

# plot all
fig = plt.figure()
ax = fig.add_subplot(111)    
fig.set_size_inches(8,6)
for i, j in enumerate(di.keys()):
    plt.fill_between(C_,np.min(log_coefs_[:,:,j],axis=0),np.max(log_coefs_[:,:,j],axis=0),lw=1,alpha=0.8,color=colors[i],label=column_names[j])
plt.xlim([1e-4,1e1])
ax.set_xscale('log')
plt.xlabel('C',fontsize=12)
plt.ylabel('Coefficient',fontsize=12)
# plt.ylim([-2,0.5])
# plt.axis('tight')
ax.legend(bbox_to_anchor=(1.05, 1.0), loc=2, borderaxespad=0.)
plt.show()
