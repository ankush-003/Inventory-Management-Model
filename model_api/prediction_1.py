import pandas as pd
from prophet import Prophet
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import warnings
from pandas.api.types import CategoricalDtype

warnings.filterwarnings("ignore")
plt.style.use('ggplot')
plt.style.use('fivethirtyeight')

# df = pd.read_csv('/content/drive/MyDrive/hashcode11/demand.csv')
# df.head()
cat_type = CategoricalDtype(categories=['Monday','Tuesday',
                                        'Wednesday',
                                        'Thursday','Friday',
                                        'Saturday','Sunday'],
                            ordered=True)

def create_features(df, label=None):
    """
    Creates time series features from datetime index.
    """
    df = df.copy()
    df['date'] = df.Date
    df['hour'] = df['date'].dt.hour
    df['dayofweek'] = df['date'].dt.dayofweek
    df['weekday'] = df['date'].dt.day_name()
    df['weekday'] = df['weekday'].astype(cat_type)
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['dayofyear'] = df['date'].dt.dayofyear
    df['dayofmonth'] = df['date'].dt.day
    df['weekofyear'] = df['date'].dt.weekofyear
    df['date_offset'] = (df.date.dt.month*100 + df.date.dt.day - 320)%1300

    df['season'] = pd.cut(df['date_offset'], [0, 300, 602, 900, 1300], 
                          labels=['Spring', 'Summer', 'Fall', 'Winter']
                   )
    X = df[['hour','dayofweek','quarter','month','year',
           'dayofyear','dayofmonth','weekofyear','weekday',
           'season']]
    if label:
        y = df[label]
        return X, y
    return X

def preprocess(df):
    df['Date'] = pd.to_datetime(df['Date'])
    category = df['Product_Category'].unique()
    category.sort()
    products = df[df['Product_Category'] == category[0]]['Product_Code'].unique()
    print(category)
    print(products)
    df = df.loc[(df['Product_Code'] == products[0]) & (df['Product_Category'] == category[0]), ['Date','Order_Demand']].reset_index(drop=True)
    # df.set_index('Date', inplace=True)
    # print(df['Product_Category'].unique().shape[0] == df.shape[0])
    df['Order_Demand'] = df['Order_Demand'].str.replace('[^0-9-]', '').apply(pd.to_numeric)
    print(df.dtypes)
    X, y = create_features(df, label='Order_Demand')
    features_target = pd.concat([X, y], axis=1)
    # features_target.head()
    # print(features_target.dtypes)
    # features_target.head()
    split_date = '22-DEC-2015'
    # df['Order_Demand'] = df['Order_Demand'].str.replace('[^0-9-]','').apply(np.int64)
    df_train = df.loc[df.Date <= split_date].copy()
    df_test = df.loc[df.Date > split_date].copy()
    train = df_train.copy().reset_index(drop=True)
    train.columns = ['ds', 'y']
    train.sort_values('ds', inplace=True)
    test = df_test.copy().reset_index(drop=True)
    test.columns = ['ds', 'y']
    test.sort_values(by='ds', inplace=True)
    return train_model(train, test)
    
    
def train_model(train, test):
    model = Prophet()
    model.fit(train)
    pred = model.predict(test)
    MAE = mean_absolute_error(test['y'], pred['yhat'])
    MSE = mean_squared_error(test['y'], pred['yhat'])
    RMSE = np.sqrt(mean_squared_error(test['y'], pred['yhat']))
    print('MAE: {}'.format(MAE))
    print('MSE: {}'.format(MSE))
    print('RMSE: {}'.format(RMSE))
    print('Mean Absolute Percentage Error: {}'.format(mean_absolute_percentage_error(y_true=test['y'], y_pred=pred['yhat'])))
    print(pred.tail())
    return [MAE, MSE, RMSE]
        
    