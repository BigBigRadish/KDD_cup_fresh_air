# -*- coding: utf-8 -*-
'''
@author:Zhukun Luo
Jiangxi university of finance and economics
'''
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
'''
from feature_process_method import  null_NO2_train_set,null_NO2_validate_set_x,null_NO2_validate_set,null_NO2_train_set1
from feature_process_method import  null_NO2_train_set_1
#空值填充
#机器学习模型填充特征值
#填充NO2的空值
null_NO2_train_set_Y=null_NO2_train_set['NO2'].get_values()
null_NO2_train_set_X=null_NO2_train_set.drop(columns=['NO2'])
null_NO2_train_set_X=null_NO2_train_set_X.get_values()

#print(null_NO2_train_set)
#lasso回归预测空值
lassocv = LassoCV(alphas=[0.01, 0.1, 0.5, 1, 3, 5, 7, 10, 20, 100], cv=5)
# 拟合训练集
lassocv.fit(null_NO2_train_set_X, null_NO2_train_set_Y.ravel())
# 打印最优的α值
#print ("最优的alpha值: "+str(lassocv.alpha_.astype('float')))
# 打印模型的系数
#print (lassocv.intercept_)
#print (lassocv.coef_)
#最优的alpha值: 100.0
#38.4427137116
test_NO2_Y_pred = lassocv.predict(null_NO2_validate_set_x)
print(list(test_NO2_Y_pred))
NO2_Y_pred=[]
for i in test_NO2_Y_pred:
    NO2_Y_pred.append(round(i))
print(NO2_Y_pred)
null_NO2_validate_set['NO2']=NO2_Y_pred
beijing_dataset_1=pd.concat([pd.DataFrame(null_NO2_train_set1),pd.DataFrame(null_NO2_validate_set),pd.DataFrame(null_NO2_train_set_1)],ignore_index=True,axis=0)#连接表，生成新的训练集
beijing_dataset_1=pd.DataFrame(beijing_dataset_1)
#print(beijing_17_18_dataset_1.describe())
#print(beijing_17_18_dataset_1.head(5))
#beijing_17_18_dataset_1.to_csv(r'../resource/beijing_17_18_dataset.csv')
#print(beijing_dataset_1.isnull().sum())
#预测SO2的残缺值
null_SO2_train_set_1=beijing_dataset_1[beijing_dataset_1['CO'].isnull()|beijing_dataset_1['NO2'].isnull()]#方便后面合并数据集
null_SO2_train_set=beijing_dataset_1[beijing_dataset_1['CO'].notnull()&beijing_dataset_1['NO2'].notnull()]
#print(null_NO2_train_set.head(10))
null_SO2_validate_set=null_SO2_train_set[null_SO2_train_set['SO2'].isnull()]#缺失的SO2验证集
null_SO2_train_set1=null_SO2_train_set[null_SO2_train_set['SO2'].notnull()]#填充SO2的训练集
null_SO2_train_set=null_SO2_train_set1.drop(columns=['PM2.5','PM10','O3'])
null_SO2_validate_set_x=null_SO2_validate_set.drop(columns=['SO2','PM2.5','PM10','O3'])
#####################################################################################
null_SO2_train_set_Y=null_SO2_train_set['SO2'].get_values()
null_SO2_train_set_X=null_SO2_train_set.drop(columns=['SO2'])
null_SO2_train_set_X=null_SO2_train_set_X.get_values()

#print(null_NO2_train_set)
#lasso回归预测空值
lassocv = LassoCV(alphas=[0.01, 0.1, 0.5, 1, 3, 5, 7, 10, 20, 100], cv=5)
# 拟合训练集
lassocv.fit(null_SO2_train_set_X, null_SO2_train_set_Y.ravel())
# 打印最优的α值
#print ("最优的alpha值: "+str(lassocv.alpha_.astype('float')))
# 打印模型的系数
#print (lassocv.intercept_)
#print (lassocv.coef_)
#最优的alpha值: 100.0
#38.4427137116
test_SO2_Y_pred = lassocv.predict(null_SO2_validate_set_x)
print(list(test_SO2_Y_pred))
SO2_Y_pred=[]
for i in test_SO2_Y_pred:
    i='{:.2f}'.format(i)#保留两位
    SO2_Y_pred.append(i)
print(SO2_Y_pred)
null_SO2_validate_set['SO2']=SO2_Y_pred
beijing_dataset_2=pd.concat([pd.DataFrame(null_SO2_train_set1),pd.DataFrame(null_SO2_validate_set),pd.DataFrame(null_SO2_train_set_1)],ignore_index=True,axis=0)#连接表，生成新的训练集
beijing_dataset_2=pd.DataFrame(beijing_dataset_2)
#####################################################################################################
#预测CO的残缺值
null_CO_train_set_1=beijing_dataset_2[beijing_dataset_2['SO2'].isnull()|beijing_dataset_2['NO2'].isnull()]#方便后面合并数据集
null_CO_train_set=beijing_dataset_2[beijing_dataset_2['SO2'].notnull()&beijing_dataset_2['NO2'].notnull()]
#print(null_NO2_train_set.head(10))
null_CO_validate_set=null_CO_train_set[null_CO_train_set['CO'].isnull()]#缺失的CO验证集
null_CO_train_set1=null_CO_train_set[null_CO_train_set['CO'].notnull()]#填充CO的训练集
null_CO_train_set=null_CO_train_set1.drop(columns=['PM2.5','PM10','O3'])
null_CO_validate_set_x=null_CO_validate_set.drop(columns=['CO','PM2.5','PM10','O3'])
#####################################################################################
null_CO_train_set_Y=null_CO_train_set['CO'].get_values()
null_CO_train_set_X=null_CO_train_set.drop(columns=['CO'])
null_CO_train_set_X=null_CO_train_set_X.get_values()

#print(null_NO2_train_set)
#lasso回归预测空值
lassocv = LassoCV(alphas=[0.01, 0.1, 0.5, 1, 3, 5, 7, 10, 20, 100], cv=5)
# 拟合训练集
lassocv.fit(null_CO_train_set_X, null_CO_train_set_Y.ravel())
# 打印最优的α值
#print ("最优的alpha值: "+str(lassocv.alpha_.astype('float')))
# 打印模型的系数
#print (lassocv.intercept_)
#print (lassocv.coef_)
#最优的alpha值: 100.0
#38.4427137116
test_CO_Y_pred = lassocv.predict(null_CO_validate_set_x)
print(list(test_CO_Y_pred))
CO_Y_pred=[]
for i in test_CO_Y_pred:
    i='{:.2f}'.format(i)#保留两位
    CO_Y_pred.append(i)
print(CO_Y_pred)
null_CO_validate_set['CO']=CO_Y_pred
beijing_dataset_3=pd.concat([pd.DataFrame(null_CO_train_set1),pd.DataFrame(null_CO_validate_set),pd.DataFrame(null_CO_train_set_1)],ignore_index=True,axis=0)#连接表，生成新的训练集
beijing_dataset_3=pd.DataFrame(beijing_dataset_3)
print(beijing_dataset_3.describe())
beijing_dataset_1.to_csv('../resource/beijing_dataset_1.csv')
'''
beijing_dataset_1=pd.read_csv(r'../resource/beijing_dataset_1.csv')
beijing_dataset_1_CNS_train=beijing_dataset_1[beijing_dataset_1['CO'].notnull()&beijing_dataset_1['NO2'].notnull()&beijing_dataset_1['SO2'].notnull()]
beijing_dataset_1_CNS_validate=beijing_dataset_1[beijing_dataset_1['CO'].isnull()&beijing_dataset_1['NO2'].isnull()&beijing_dataset_1['SO2'].isnull()]
#print(beijing_dataset_1_1.describe())
#print(beijing_dataset_1_CNS_train)太少，还是应该填补缺失值
'''
beijing_dataset_1_CNS_train_1=beijing_dataset_1_CNS_train.drop( columns=['PM2.5','PM10','O3'])

划分训练集

target=beijing_dataset_1_CNS_train_1['CO']
# 避免过拟合，采用交叉验证，验证集占训练集20%，固定随机种子（random_state)  
train_X,test_X, train_y, test_y = train_test_split(beijing_dataset_1_CNS_train_1.drop(columns=['CO','NO2','SO2']),  
                                                   target,  
                                                   test_size = 0.2,  
 
                                                   random_state = 0)  
太慢了
#遗传 选最优算法模型，最佳特征
tpot = TPOTRegressor(generations=10, verbosity=2) #迭代150次  
tpot.fit(train_X, train_y)  
print(tpot.score(test_X, test_y))  
tpot.export('pipeline.py') 


#params = [1]
#test_scores = []
#for param in params:
    #clf = XGBRegressor( learning_rate=0.1, max_depth=param, min_child_weight=2)#效果一搬 #56%max_depth=1
#     clf=GradientBoostingRegressor(loss='ls', alpha=0.9,#40%
#                                             n_estimators=500,
#                                             learning_rate=0.05,
#                                             max_depth=1,
#                                             subsample=0.8,
#                                             min_samples_split=9,
#                                             max_leaf_nodes=10)#这方法也不行
las=LassoCV(alphas=[100], cv=5)#接近70%
lr = LinearRegression()#效果不好不到50%
rfg = RandomForestRegressor(bootstrap=True, max_features=0.005, min_samples_leaf=11, min_samples_split=10,
                                        n_estimators=100)#最高59%

svr_rbf = SVR(kernel='rbf')
clf=svr_rbf#跑不出来。。。我擦
stregr = StackingRegressor(regressors=[las, las, las, las], meta_regressor=svr_rbf)
test_score = np.sqrt(-cross_val_score(stregr, train_X, train_y, cv=10, scoring='neg_mean_squared_error'))#70%的正确率,与lasso差不多
print(test_score)    
#pl.plot(params, test_scores)
#pl.title("max_depth vs CV Error");
#pl.show()
#此路不通
'''
print(beijing_dataset_1_CNS_train.isnull().sum())
beijing_dataset_2=beijing_dataset_1_CNS_train[beijing_dataset_1_CNS_train['PM2.5'].notnull()&beijing_dataset_1_CNS_train['PM10'].notnull()&beijing_dataset_1_CNS_train['O3'].notnull()]
beijing_dataset_2.to_csv(r'../resource/beijing_dataset2.csv')
print(beijing_dataset_2)
# 避免过拟合，采用交叉验证，验证集占训练集20%，固定随机种子（random_state) 
target=beijing_dataset_2['O3'] 
train_X,test_X, train_y, test_y = train_test_split(beijing_dataset_2.drop(columns=['pressure']),  
                                                   target,  
                                                   test_size = 0.2,  
 
                                                   random_state = 0)
svr_rbf = SVR(kernel='rbf')
test_score = np.sqrt(-cross_val_score(svr_rbf, test_X, test_y, cv=10, scoring='neg_mean_squared_error'))#
print(test_score)    
#pl.plot(params, test_scores)
#pl.title("max_depth vs CV Error");
#pl.show()