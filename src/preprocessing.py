import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def preprocess_train(csv_path="../data/processed/train.csv", lookback=24):
    
    #load data
    train = pd.read_csv(csv_path, index_col= 0, parse_dates=True)

    #ensure time order
    train = train.sort_index()

    #drop nan values
    train = train.dropna()
    
    #separate feature and target
    x = train.drop(columns=['pm2.5'])
    y = train['pm2.5']

    #scale 
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    x_scaled = scaler_x.fit_transform(x)
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

    #create a sequence
    x_seq, y_seq = [],[]

    for i in range(len(x_scaled) - lookback):
        x_seq.append(x_scaled[i:i+ lookback])
        y_seq.append(y_scaled[i+ lookback])

    x_seq = np.array(x_seq)
    y_seq = np.array(y_seq)

    return x_seq, y_seq, scaler_x, scaler_y

def preprocess_test(csv_path= "../data/processed/test.csv", lookback=24, scaler_x= None):

    #load data
    test = pd.read_csv(csv_path, index_col=0,parse_dates=True)

    #set datetime in order
    tets = test.sort_index()

    #drop non values
    test = test.dropna()

    # features
    x = test

    #scale feature using traain scaler_x

    x_scaled = scaler_x.transform(x)

    # creat sequences
    x_seq = []

    for i in range(len(x_scaled) - lookback):
        x_seq.append(x_scaled[i:i + lookback])

    x_seq = np.array(x_seq)

    return x_seq, test.index[lookback:]