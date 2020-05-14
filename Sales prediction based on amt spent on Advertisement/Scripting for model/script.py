# -*- coding: utf-8 -*-
"""
Created on Mon May 11 14:46:56 2020

@author: GSLP0676
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from  sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.externals import joblib

#Load data from csv
data=pd.read_csv('Advertising.csv')
data.head(5)

#Dropping of unnecessary columns
data.drop(columns='Unnamed: 0',inplace=True,axis=1)

#Checking for null values
data.isna().sum()

#Visualizing distribution of data
for columns in data:
    p=1
    if p<=3:
        plt.figure(figsize=(3,5), facecolor='white')
        ax = plt.subplot()
        sns.distplot(data[columns])
        plt.show()
        p+=1
    
#Taking features(X) and prediction(Y) columns    
X=data.drop(columns='sales')        
y=data['sales']  

#Standard Scaling of data
scaler = StandardScaler()
scaler.fit_transform(X)

#Splitting data into train and test
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
X_train.sort_index(axis=1, inplace=True)
X_test.sort_index(axis=1,inplace=True)
X_train.head(2)
X_test.head(2)
X_test.info()


#Random Forest Regressor
ranreg = RandomForestRegressor()
ranreg.fit(X_train,y_train)
ranreg.score(X_test,y_test)

#XGBoos Regressor
xgbreg=XGBRegressor()
xgbreg.fit(X_train,y_train)
xgbreg.score(X_test,y_test)

#Hyperparameter tuning for Random forest regressor by randomized search
rand_param={ 
             'criterion': ['mse','mae'] ,
              'max_depth': range(2,50,5),
               'max_features' :['auto','sqrt','log2'] ,
               'max_leaf_nodes': range(8,80,5),
               'min_samples_leaf': range(1,12,1),
               'min_samples_split': range(2,12,1),
               'n_estimators': range(20,100,10)
               }



rand_search = RandomizedSearchCV(estimator=ranreg,param_distributions=rand_param,verbose=1,n_jobs=-1,n_iter=200,cv=7)
rand_search.fit(X_train,y_train)
best_param=rand_search.best_params_
best_param

ranreg = RandomForestRegressor(n_estimators=90,min_samples_split=3,min_samples_leaf=2,max_leaf_nodes=38,max_features='auto',max_depth=7,criterion='mse')
ranreg.fit(X_train,y_train)
ranreg.score(X_test,y_test)


#Hyperparameter tuning for XGBoost by randomized search
rand_param_xgb={
            'n_estimators':range(100,300,50),
            'eta':[0.1,0.2],
            'max_depth':range(3,10,1),
            'subsample':[0.5,0.6,0.7,0.8,0.9,1.0]
              }
rand_search_xgb = RandomizedSearchCV(estimator=xgbreg,param_distributions=rand_param_xgb,verbose=1,n_jobs=-1,n_iter=200,cv=8)
rand_search_xgb.fit(X_train,y_train)
best_param=rand_search_xgb.best_params_
best_param

xgbreg=XGBRegressor(subsample=0.5,n_estimators=150,max_depth=4,eta=0.1)
xgbreg.fit(X_train,y_train)
xgbreg.score(X_test,y_test)   

#Light GBM REgressor
lgbm=LGBMRegressor()
lgbm.fit(X_train,y_train)
lgbm.score(X_test,y_test)
lgbm.score(X_train,y_train)

X_train.columns
#Saving models
joblib.dump(ranreg,'RFReg_model.ml')
joblib.dump(xgbreg,'XGBReg_model.ml')
joblib.dump(lgbm,'LGBMReg_model.ml')


