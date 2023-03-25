import pandas as pd
from prophet import Prophet
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import warnings
from datetime import datetime

def create_feature(dataframe):
    dataframe = dataframe.copy()
    dataframe['day_of_the_week'] = dataframe.index.dayofweek
    dataframe['Quarter'] = dataframe.index.quarter
    dataframe['Month'] = dataframe.index.month
    dataframe['Year'] = dataframe.index.year
    dataframe['Week'] = dataframe.index.isocalendar().week.astype(int)
    return dataframe

def preprocess(df, BU, PF, PLID):
    df.columns = ['BU','PF','PLID','FQ','FM','Order_Demand','Date']
    df = df.drop(['FQ', 'FM'], axis=1)
    # print(df)
    df = df[df['BU'] == BU]
    df = df[df['PF'] == PF]
    df = df[df['PLID'] == PLID]
    df.drop(['BU','PF','PLID'], axis=1, inplace=True)
    #df['Order_Demand'] = df11['Order_Demand'].str.replace('[^0-9-]','').apply(np.int64)
    df['Date'] = pd.to_datetime(df['Date'] ,format='%d-%m-%Y')
    df.sort_values(by='Date', inplace = True)
    df11 = df.groupby('Date')['Order_Demand'].sum().reset_index()
    trim_date = datetime(2021, 12, 31)
    df11 = df11[df11['Date'] <= trim_date]
    df11.set_index('Date', inplace=True)
    df = create_feature(df11)
    # features, Target variable
    Features = ['day_of_the_week', 'Quarter','Month', 'Year', 'Week']
    target = ['Demand']
    df_month = df.resample('MS').mean()
    df_Q = df.resample('Q').mean()
    train_end = datetime(2020, 12, 31)
    test_end = datetime(2021, 12, 31)

    # df_train = df_Q[:train_end]
    # df_test = df_Q[train_end:test_end]
    df_train = df_month[:train_end]
    df_test = df_month[train_end:test_end]
    df_train.drop(['day_of_the_week', 'Quarter','Month','Year','Week'], axis=1, inplace=True)
    df_test.drop(['day_of_the_week', 'Quarter','Month','Year','Week'], axis=1, inplace=True)
    df_train = df_train.reset_index()
    df_test = df_test.reset_index()
    
    df_train.columns = ['ds','y']
    df_test.columns = ['ds','y']
    print("range is :")
    print(min(df_train['y'].min(), df_test['y'].min()) - max(df_train['y'].max(), df_test['y'].max()))
    
    m = Prophet()
    m.fit(df_train)
    pred = m.predict(df_test)
    df_pred = pred[['ds', 'yhat']]
    
    MSE = mean_squared_error(df_test['y'], df_pred['yhat'])
    RMSE = np.sqrt(MSE)
    MAE = mean_absolute_error(df_test['y'], df_pred['yhat'])
    MAPE = mean_absolute_percentage_error(df_test['y'], df_pred['yhat'])
    ds = [ i.strftime('%Y-%m-%d') for i in df_test['ds']]
    y = [ i for i in df_test['y']]
    yhat = [ i for i in df_pred['yhat']]
    return [MSE, MAE, RMSE, MAPE, ds, y, yhat]


