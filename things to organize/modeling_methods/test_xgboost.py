# -*- coding: utf-8 -*-
"""
XGBoost

XGboost installation from binaries:
http://www.picnet.com.au/blogs/guido/post/2016/09/22/xgboost-windows-x64-binaries-for-download/

Used git bash to run git clone command,
then anaconda prompt to run python setup.py install
"""

##############################################################################

import sys
sys.path.append(r'C:\Users\jb2428\Desktop\python\whi')

from analytic_toolkit import simulate_two_class

X_train, X_test, y_train, y_test = simulate_two_class([200,150])


##############################################################################

import xgboost as xgb
from sklearn.preprocessing import LabelBinarizer

lb = LabelBinarizer()
y_train_ = lb.fit_transform(y_train)
y_test_ = lb.transform(y_test)

dtrain = xgb.DMatrix(X_train.values, label=y_train_)
dtest = xgb.DMatrix(X_test.values, label=y_test_)


param = {'bst:max_depth':2, 'bst:eta':1, 'silent':1, 'objective':'binary:logistic', 'nthread':4, 'eval_metric':'auc'}

evallist  = [(dtest,'eval'), (dtrain,'train')]

num_round = 10
bst = xgb.train(param, dtrain, num_round, evallist)


print('Completed Training')


xgb.plot_importance(bst)
# xgb.plot_tree(bst, num_trees=2)