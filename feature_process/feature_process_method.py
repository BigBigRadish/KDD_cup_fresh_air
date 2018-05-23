# -*- coding: utf-8 -*-
'''
@author:Zhukun Luo
Jiangxi university of finance and economics
'''
import pandas as pd
import numpy as np
import sklearn
import re
from sklearn.preprocessing import  OneHotEncoder#onehot对独立型特征进行编码
from sklearn.preprocessing import MinMaxScaler#缩放特征值
from sklearn.preprocessing import Normalizer#特征归一化
from pandas.tests.io.parser import skiprows, index_col
#https://www.cnblogs.com/DjangoBlog/p/7794832.html
#经纬缩放并归一
#月日具体时间onehot（）
#pressure归一
#wind_dirict归一
#数据都归一化
#beijing_history_grid=pd.read_csv(r'../resource\Beijing_historical_meo_grid.csv')
'''17-18数据合并
beijing_17_18_meo=pd.read_csv(r'../resource/beijing_17_18_meo.csv',skiprows=1,index_col=False,names=['station_id','longitude','latitude','utc_time','temperature','pressure','humidity','wind_direction','wind_speed','weather'] ,low_memory=False) 
print(beijing_17_18_meo.describe())
beijing_17_18_aq=pd.read_csv(r'../resource/beijing_17_18_aq.csv',skiprows=1,index_col=False ,names=['station_id','utc_time','PM2.5','PM10','NO2','CO','O3','SO2'],low_memory=False)
print(beijing_17_18_aq.describe())
list=[]                     
for index, row in beijing_17_18_meo.iterrows():
    #print(str(row['station_id']).replace('_meo', ''))
    list.append(str(row['station_id']).replace('_meo', ''))
beijing_17_18_meo['station_id']=list
#beijing_17_18_meo.set_index(beijing_17_18_meo['unique_feature'])
list1=[]
for index, row in beijing_17_18_aq.iterrows():
    list1.append(str(row['station_id']).replace('_aq', ''))
beijing_17_18_aq['station_id']=list1
beijing_17_18_dataset=pd.merge(beijing_17_18_meo,beijing_17_18_aq,how='left', on=['station_id','utc_time'])                     
beijing_17_18_dataset.to_csv(r'../resource/train_dataset.csv')

#2.3月份数据合并
beijing_02_03_me=pd.read_csv(r'../resource/beijing_201802_201803_me.csv',skiprows=1,index_col=False,names=['station_id','utc_time','weather','temperature','pressure','humidity','wind_speed','wind_direction'] ,low_memory=False) 
print(beijing_02_03_me.describe())
beijing_02_03_aq=pd.read_csv(r'../resource/beijing_201802_201803_aq.csv',skiprows=1,index_col=False ,names=['station_id','utc_time','PM2.5','PM10','NO2','CO','O3','SO2'],low_memory=False)
print(beijing_02_03_aq.describe())
list=[]                     
for index, row in beijing_02_03_me.iterrows():
    #print(str(row['station_id']).replace('_meo', ''))
    list.append(str(row['station_id']).replace('_meo', ''))
beijing_02_03_me['station_id']=list
#beijing_17_18_meo.set_index(beijing_17_18_meo['unique_feature'])
list1=[]
for index, row in beijing_02_03_aq.iterrows():
    list1.append(str(row['station_id']).replace('_aq', ''))
beijing_02_03_aq['station_id']=list1
beijing_02_03_dataset=pd.merge(beijing_02_03_me,beijing_02_03_aq,how='left', on=['station_id','utc_time'])                     



beijing_17_18_dataset=pd.read_csv(r'../resource/train_dataset.csv',index_col=False)
#print(beijing_17_18_dataset.isnull().sum()) 
#给02-03表添加经度，纬度
latitude_02_03=[]
longitude_02_03=[]
beijing_station_id_longitude_latitude=beijing_17_18_dataset[["station_id",'latitude','longitude']]
beijing_station_id_longitude_latitude=beijing_station_id_longitude_latitude.drop_duplicates(subset=['station_id'], keep='first')   
beijing_station_id_longitude_latitude.to_csv(r'../resource/station_latitude_longitude.csv')#经纬度信息
for index,i in beijing_station_id_longitude_latitude.iterrows():
    for index,  j in beijing_02_03_dataset.iterrows():
        if (str(j['station_id'])==str(i['station_id'])):
            latitude_02_03.append(i['latitude'])
            longitude_02_03.append(i['longitude'])
beijing_02_03_dataset['latitude']=latitude_02_03
beijing_02_03_dataset['longitude']=longitude_02_03
beijing_02_03_dataset.to_csv(r'../resource/02_03_train_dataset.csv')
'''
'''
beijing_17_18_dataset=pd.read_csv(r'../resource/train_dataset.csv',index_col=False)
beijing_02_03_dataset=pd.read_csv(r'../resource/02_03_train_dataset.csv',index_col=False)
beijing_dataset=pd.concat([beijing_17_18_dataset,beijing_02_03_dataset],ignore_index=True,axis=0)#连接表，生成新的训练集
beijing_dataset.to_csv('../resource/beijing_dataset.csv')#总数据
'''
beijing_dataset=pd.read_csv('../resource/beijing_dataset.csv',index_col=False)
#将时间中的月份和具体时间提取出来，形成新的特征
month=[]
day=[]
time=[]
for i in beijing_dataset['utc_time']:
    month.append(re.findall(r'/(.*)/', str(i))[0])
    day.append(re.findall(r'/([0-9]{0,2}) ', str(i))[0]) 
    time.append(re.findall(r' (.*):00', str(i))[0])
beijing_dataset['month']=month
beijing_dataset['day']=day
beijing_dataset['time']=time
nullvalue_train_set=beijing_dataset[['longitude','latitude','month','day','time','temperature','pressure','humidity','wind_direction','wind_speed','weather','NO2','CO','SO2','PM2.5','PM10','O3']]
nullvalue_train_set.to_csv(r'../resource/expand_train_set')        
#print(nullvalue_train_set.isnull().sum()) 
#longitude 和 latitude特征缩放
train_set_longitude=MinMaxScaler(feature_range=(0,10)).fit_transform(nullvalue_train_set['longitude'].reshape(-1, 1))
print(train_set_longitude)
nullvalue_train_set['longitude']=train_set_longitude
train_set_latitude=MinMaxScaler(feature_range=(0,10)).fit_transform(nullvalue_train_set['latitude'].reshape(-1, 1))
nullvalue_train_set['latitude']=train_set_latitude
train_set_pressure=MinMaxScaler(feature_range=(0,10)).fit_transform(nullvalue_train_set['pressure'].reshape(-1, 1))
nullvalue_train_set['pressure']=train_set_pressure
#处理特殊值wind_direction
wind_direction=[]
for i in nullvalue_train_set['wind_direction'].get_values():
    if(i!=999017):
        wind_direction.append(i)
    else:
        i=0
        wind_direction.append(i)
nullvalue_train_set['wind_direction']=wind_direction  
#特征编码
nullvalue_train_set=pd.get_dummies(nullvalue_train_set,columns=['month','day','time','weather']) 
nullvalue_train_set=nullvalue_train_set.astype('float')  
#print(nullvalue_train_set.describe())     
#训练集
nullvalue_train_set=nullvalue_train_set[nullvalue_train_set['wind_direction'].notnull()&nullvalue_train_set['wind_speed'].notnull()]#去掉丢失的wind数据234行
train_set_wind_direction=MinMaxScaler(feature_range=(0,10)).fit_transform(nullvalue_train_set['wind_direction'].reshape(-1, 1))#风向缩放
nullvalue_train_set['wind_direction']=train_set_wind_direction
#print(nullvalue_train_set.isnull().sum()) 
#nullvalue_train_set.to_csv(r'../resource/train_scaler_onehot_set.csv')
#PM_O3=nullvalue_train_set[['PM2.5','PM10','O3']]
#nullvalue_train_set=nullvalue_train_set.drop(columns=['PM2.5','PM10','O3'])
#预测NO2空值
null_NO2_train_set_1=nullvalue_train_set[nullvalue_train_set['CO'].isnull()|nullvalue_train_set['SO2'].isnull()]#方便后面合并数据集
null_NO2_train_set=nullvalue_train_set[nullvalue_train_set['CO'].notnull()&nullvalue_train_set['SO2'].notnull()]
#print(null_NO2_train_set.head(10))
null_NO2_validate_set=null_NO2_train_set[null_NO2_train_set['NO2'].isnull()]#缺失的NO2验证集
null_NO2_train_set1=null_NO2_train_set[null_NO2_train_set['NO2'].notnull()]#填充NO2的训练集
null_NO2_train_set=null_NO2_train_set1.drop(columns=['PM2.5','PM10','O3'])
null_NO2_validate_set_x=null_NO2_validate_set.drop(columns=['NO2','PM2.5','PM10','O3'])
#print(null_NO2_validate_set.head(10))
