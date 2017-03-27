# -*- coding: utf-8 -*-
"""

"""

import pandas as pd
import numpy as np
from scipy.stats import skew
from sklearn.linear_model import LassoCV
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import cross_val_score

# load the train and test data
house_train=pd.read_csv('train.csv')
house_test=pd.read_csv('test.csv')
house_train=house_train.loc[house_train['GrLivArea']<4000,]
house_train=house_train.loc[house_train['LotArea'] <100000,]
house_test.loc[1132,'GarageYrBlt']=2007

all_data =  pd.concat((house_train.loc[:,'MSSubClass':'SaleCondition'],
                      house_test.loc[:,'MSSubClass':'SaleCondition']),
                        ignore_index=True)
                      
#all_data.drop('GarageYrBlt',axis=1)
def change_BedroomAbvGr(number):
    if number >=6:
        return 6
    else:
        return number
all_data['BedroomAbvGr']=all_data['BedroomAbvGr'].apply(change_BedroomAbvGr)
def change_TotRmsAbvGrd(number):
    if number >=11:
        return 11
    else:
        return number
all_data['TotRmsAbvGrd']=all_data['TotRmsAbvGrd'].apply(change_TotRmsAbvGrd)
def change_KitchenAbvGr(number):
    if number>=2:
        return 2
    else:
        return number
all_data['KitchenAbvGr']=all_data['KitchenAbvGr'].apply(change_KitchenAbvGr)  

def change_LotShape(cat):
    if cat=='IR2' or cat=='IR3':
        return 'IR2'
    else:
        return cat
all_data['LotShape']=all_data['LotShape'].apply(change_LotShape)

def change_GarageQual(cat):
    if cat=='Po' or cat=='Fa' or cat=='NO':
        return 'Po'
    else:
        return cat
all_data['GarageQual']=all_data['GarageQual'].fillna('NO')    
all_data['GarageQual']=all_data['GarageQual'].apply(change_GarageQual)
all_data['MSSubClass']=all_data['MSSubClass'].replace(
        {20: 1, 30: 0, 40: 0, 45: 0,50: 0, 60: 1, 70: 0, 75: 0, 80: 0, 85: 0,
         90: 0, 120: 1, 150: 0, 160: 0, 180: 0, 190: 0})
#all_data.drop('YrSold',axis=1)

all_data["HighSeason"] = all_data["MoSold"].replace( 
        {1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 1, 7: 1, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0})

all_data.drop('MoSold',axis=1) 

all_data.drop('BsmtFinSF2',axis=1)
all_data['BsmtUnfSF_med']=(all_data['BsmtUnfSF']>all_data['BsmtUnfSF'].median())*1
def change_HeatingQC(cat):
    if cat=='Po' or cat=='Fa':
        return 'Po'
    else:
        return cat
    
all_data['HeatingQC']=all_data['HeatingQC'].apply(change_HeatingQC)      
all_data["Fence"] = all_data["Fence"].map(
        {None: 0, "MnWw": 1, "GdWo": 1, "MnPrv": 1, "GdPrv": 1}).astype(int)  
all_data["HasEnclosedPorch"] = (all_data["EnclosedPorch"] == 0) * 1
all_data.drop('EnclosedPorch',axis=1)

all_data['Exterior1st']=all_data['Exterior1st'].fillna('VinylSd')

#all_data['ScreenPorch']=np.log1p(all_data['ScreenPorch'])

all_data['ScreenPorch_if0']=(all_data['ScreenPorch'] !=0)*1

all_data.drop('ScreenPorch',axis=1)
all_data['Exterior2nd']=all_data['Exterior2nd'].fillna('VinylSd')       
all_data['Functional']=all_data['Functional'].fillna('Typ')       
all_data['SaleType']=all_data['SaleType'].fillna('WD')
house_train["SalePrice"] = np.log1p(house_train["SalePrice"])

'''
numeric_features = all_data.dtypes[all_data.dtypes != "object"].index

skewed_features = house_train[numeric_features].apply(lambda x: skew(x.dropna()))

# when skewed > 0.75, we use log to normalized the data

skewed_features = skewed_features[skewed_features > 0.8]
skewed_features = skewed_features.index
'''
skewed_features=['1stFlrSF','LotFrontage','OpenPorchSF','GrLivArea','LotArea','WoodDeckSF',
                 'MasVnrArea']
all_data[skewed_features] = np.log1p(all_data[skewed_features])

cate_features=all_data.dtypes[all_data.dtypes == "object"].index

# fill out the Missing value
for col in cate_features:
    all_data[col]=all_data[col].fillna('NO')
all_data = all_data.fillna(all_data.median())
all_data = pd.get_dummies(all_data)

X_train = all_data[:house_train.shape[0]]
X_test = all_data[house_train.shape[0]:]
y = house_train.SalePrice

# the estimators for GridSearchCV
model_lasso = LassoCV(alphas = [0.0004],max_iter=100000)

# use bagging with linear model
bagging=BaggingRegressor(base_estimator=model_lasso,n_estimators=190).fit(X_train,y)

def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, X_train, y, scoring="neg_mean_squared_error", cv =5))
    return rmse
    
   
print rmse_cv(bagging).mean() # 0.12234

pred=bagging.predict(X_test)
pred_df=pd.DataFrame(np.expm1(pred),index=house_test['Id'])
#pred_df.to_csv('pred_day0227version2.csv')
