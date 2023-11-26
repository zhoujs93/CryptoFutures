import lightgbm as lgb
import pandas as pd
import numpy as np
from itertools import islice
import tscv
from joblib import dump,load
from models import *
# from tensorflow.keras.callbacks import Callback, ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from tensorflow.python.keras.callbacks import Callback, ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
import tensorflow_addons as tfa

def generate_label(df, threshold = 0.002):
    df['label'] = 0
    df.loc[(df['target_15m'] <= -1*threshold), 'label'] = 1
    df.loc[(df['target_15m'] >= threshold), 'label'] = 2
    return df

def get_na_features(df):
    tmp = pd.DataFrame(df[train_features].isnull().sum())
    tmp = tmp[tmp[0] > 0].reset_index()
    tmp.columns = ['feat', 'cnt']
    tmp = tmp.sort_values('cnt')
    feat_groups = dict(tmp.groupby('cnt')['feat'].agg(lambda x: list(x)))
    return feat_groups

def normalize_float_columns(df, features):
  float_cols = df[features].select_dtypes(exclude = [float]).columns
  df[float_cols] = (df[float_cols] - df[float_cols].mean()) / df[float_cols].std()
  return df

class Params: pass
param = Params()

if __name__ == '__main__':
    train = False
    df = pd.read_feather('../data/df_btc_eth_with_features.feather')
    cols_to_drop = ['open_time', 'close_time', 'ignore',
                    'create_time', 'symbol', 'returns', 'returns_5m',
                    'open', 'high', 'low', 'close', 'target_15m', 'label',
                    'sum_open_interest', 'sum_open_interest_value',
                    'count_toptrader_long_short_ratio',
                    'sum_toptrader_long_short_ratio',
                    'count_long_short_ratio', 'sum_taker_long_short_vol_ratio']

    df = df.sort_values(by='open_time', ignore_index=True)
    df = generate_label(df, threshold=0.002)

    start_time = df['open_time'].min()
    end_time = df['open_time'].max()
    dates = df['open_time'].unique()
    n = len(dates)
    train_idx = int(0.7 * n)
    valid_idx = int(0.2 * n)
    train_end = dates[train_idx]
    valid_end = dates[valid_idx]

    train_df = df.loc[df['open_time'] < train_end].reset_index(drop=True)
    valid_df = df.loc[(train_end <= df['open_time']) & (df['open_time'] < valid_end)].reset_index(drop=True)
    test_df = df.loc[(df['open_time'] >= valid_end)].reset_index(drop=True)

    groups = pd.factorize(
        train_df['open_time'].dt.day.astype(str) + '_' + train_df['open_time'].dt.month.astype(str) + '_' + train_df[
            'open_time'].dt.year.astype(str))[0]

    cv = tscv.PurgedGroupTimeSeriesSplit(
        n_splits=5,
        group_gap=31,
    )

    train_features = [x for x in df.columns if (x not in cols_to_drop)]

    train_df['token'] = train_df['token'].astype('category').cat.codes
    object_cols = train_df[train_features].select_dtypes(include=object).columns
    train_df[object_cols] = train_df[object_cols].astype(float)

    nan_features = get_na_features(train_df)
    grouped_train = train_df.groupby(['token'])
    for k, v in nan_features.items():
        for value in v:
            train_df[value] = grouped_train[value].transform(lambda x: x.ffill().fillna(0.0))

    feature_cols = pd.DataFrame(train_features)
    dtype_df = pd.DataFrame(train_df[train_features].select_dtypes(exclude=[float]).columns)
    train_features = [x for x in train_features if x not in dtype_df.values]
    train_features_test = train_features[:5]

    params = {'num_columns': len(train_features_test),
              'num_labels': 3,
              'hidden_units': [96, 96, 896, 448, 448, 256],
              'dropout_rates': [0.03527936123679956, 0.038424974585075086, 0.42409238408801436, 0.10431484318345882,
                                0.49230389137187497, 0.32024444956111164, 0.2716856145683449, 0.4379233941604448],
              'ls': 0,
              'lr': 1e-3,
              }

    ### model parameters
    param.layers = [500, 350, 200]
    param.dropout_rate = 0.35

    ###training parameters
    param.bs = 8192
    param.lr = 0.002
    param.epochs = 30
    param.wd = 0.02

    ### adding overall AuC as a metric
    ### for early stopping I only look at resp and resp_sum because they start overfitting earlier
    metrics = [tf.keras.metrics.SparseCategoricalAccuracy(name='ACC')]

    scores = []
    batch_size = 4096
    train_features_test = train_features[115:120]
    test_df = train_df[train_features_test]


    for fold, (train_idx, val_idx) in enumerate(cv.split(train_df, train_df['label'], groups)):
        min_train, max_train = min(train_df['open_time'].iloc[train_idx]).to_pydatetime(), max(
            train_df['open_time'].iloc[train_idx]).to_pydatetime()
        min_valid, max_valid = min(train_df['open_time'].iloc[val_idx]).to_pydatetime(), max(
            train_df['open_time'].iloc[val_idx]).to_pydatetime()
        x_train, x_val = train_df[train_features_test].iloc[train_idx], train_df[train_features_test].iloc[
            val_idx]
        print(f'{fold} : Train Date is from {min_train} - {max_train}')
        print(f'{fold} : Valid Date is from {min_valid} - {max_valid}')
        y_train, y_val = train_df['label'].iloc[train_idx], train_df['label'].iloc[val_idx]
        x_train = normalize_float_columns(x_train, train_features_test)
        x_val = normalize_float_columns(x_val, train_features_test)
        print(f'{fold} : Train Date is from {min_train} - {max_train}')
        print(f'{fold} : Valid Date is from {min_valid} - {max_valid}')
        if fold == 0:
            print(f'Shape of Xtrain is {x_train.shape}, Shape of yTrain is {y_train.shape}')
        ckp_path = f'../output/MLP_{fold}.hdf5'
        model = create_model(len(train_features_test), 3, params.layers, params.dropout_rate,
                             optimizer=tfa.optimizers.Lookahead(
                                 tfa.optimizers.LAMB(learning_rate=params.lr, weight_decay_rate=params.wd)
                             ),
                             metrics=metrics)
        cbs = [tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3,
                                                    patience=3, verbose=1),
               tf.keras.callbacks.EarlyStopping(
                   monitor='val_loss', patience=5, verbose=1,
                   mode='min', restore_best_weights=True
               )
               ]

        model.fit(x_train.values, y_train.values, validation_data=(x_val.values, y_val.values),
                  epochs=params.epochs,
                  batch_size=params.bs, validation_batch_size=500_000,
                  callbacks=[cbs], verbose=2)

    # 1 - 40,  20 - 60,  40 - 80,  60 - 100, 80 - 120, 100 - 140,
    # 41 - 81, 61 - 101, 81 - 121, 101 - 141, 121 - 161, 141 - 181,