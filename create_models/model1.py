# -*- coding: utf-8 -*-
'''
@author:Zhukun Luo
Jiangxi university of finance and economics
'''
#beijing model
import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.cross_validation import train_test_split 
from tpot import TPOTRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR 
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from mlxtend.regressor import StackingRegressor
import matplotlib.pyplot as pl 
beijing_dataset_2=pd.read_csv(r'../resource/beijing_dataset2.csv')
print(beijing_dataset_2)
########################################################O3########################################################
# 避免过拟合，采用交叉验证，验证集占训练集20%，固定随机种子（random_state) 
target=beijing_dataset_2['O3'] 
train_X,test_X, train_y, test_y = train_test_split(beijing_dataset_2.drop(columns=['pressure','O3']),  
                                                   target,  
                                                   test_size = 0.2,  
 
                                                   random_state = 0)

lassocv = LassoCV(alphas=[0.01, 0.1, 0.5, 1, 3, 5, 7, 10, 20, 100], cv=5)#0.329379497536
# 拟合训练集
#lassocv.fit(train_X, train_y)
# 打印最优的α值
#print ("最优的alpha值: "+str(lassocv.alpha_.astype('float')))
# 打印模型的系数
#print (lassocv.intercept_)
#print (lassocv.coef_)
rfg = RandomForestRegressor(bootstrap=True, max_features=0.005, min_samples_leaf=11, min_samples_split=10,#0.459946225678
                                        n_estimators=100)
svr_rbf = SVR(kernel='rbf')
lr=LinearRegression()#0.70
xgb= XGBRegressor()#0.850285010253
stregr = StackingRegressor(regressors=[xgb, lr], meta_regressor=svr_rbf)#0.248584836784
#rfg = RandomForestRegressor(bootstrap=True, max_features=0.005, min_samples_leaf=11, min_samples_split=10,
#                                        n_estimators=100)#最高59%
#xgb.fit(train_X,train_y)#8.93490919332e-05
#predict_y=xgb.predict(test_X)
xgb.fit(train_X,train_y)
predict_y=xgb.predict(test_X)
print(xgb.score(test_X,test_y))
test_score = cross_val_score(xgb, train_X, train_y, cv=10, scoring='neg_mean_squared_error')#
print(test_score)    
from sklearn.metrics import mean_squared_log_error,r2_score
#print(mean_squared_log_error(test_y, predict_y))
#print(r2_score(test_y, predict_y,multioutput='variance_weighted'))
#print(r2_score(test_y, predict_y,multioutput='uniform_average'))
#rfg   
#pl.plot(params, test_scores)
#pl.title("max_depth vs CV Error");
#pl.show()
#最后选择线性模型
#预测PM10#####################################################################################################################
target1=beijing_dataset_2['PM10']#0.792095409402 
train_X,test_X, train_y, test_y = train_test_split(beijing_dataset_2.drop(columns=['pressure','PM10']),  
                                                   target1,  
                                                   test_size = 0.2,  
 
                                                   random_state = 0)
xgb.fit(train_X,train_y)
print(xgb.score(test_X,test_y))
print(xgb.predict(test_X))
print(test_y)
################################################################pm2.5####################################################
target1=beijing_dataset_2['PM2.5'] #0.895274815606
train_X,test_X, train_y, test_y = train_test_split(beijing_dataset_2.drop(columns=['pressure','PM2.5']),  
                                                   target1,  
                                                   test_size = 0.2,  
 
                                                   random_state = 0)
xgb.fit(train_X,train_y)
print(xgb.score(test_X,test_y))
print(xgb.predict(test_X))
print(test_y)