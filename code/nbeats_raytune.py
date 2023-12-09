import numpy as np
import pandas as pd
from kerasbeats import prep_time_series, NBeatsModel
import tensorflow as tf
from ray import train, tune
from tensorflow.python.keras.callbacks import Callback, ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from ray.train import ScalingConfig
from ray.train.tensorflow import TensorflowTrainer
from ray.data import read_numpy
from ray.train.tensorflow.keras import ReportCheckpointCallback
from ray.tune import Tuner, with_resources
from ray.tune.schedulers import HyperBandScheduler
from ray.tune.search.bayesopt import BayesOptSearch
from sklearn.preprocessing import MinMaxScaler


def train_func(config, data):
    batch_size = 1024
    num_generic_neurons = config['num_generic_neurons']
    num_generic_stacks = config['num_generic_stacks']
    num_generic_layers = config['num_generic_layers']
    num_trend_neurons = config['num_trend_neurons']
    num_trend_stacks = config['num_trend_stacks']
    num_trend_layers = config['num_trend_layers']
    num_seasonal_neurons = config['num_seasonal_neurons']
    polynomial_term = config['polynomial_term']
    lr = 1e-3

    model = NBeatsModel(**config)
    model.build_layer()
    model.build_model()
    # ckp = ModelCheckpoint(ckp_path, monitor='r2score', verbose=0,
    #                       save_best_only=True, save_weights_only=True, mode='max')
    es = EarlyStopping(monitor='r2score', min_delta=1e-4, patience=10, mode='max',
                       baseline=None, restore_best_weights=True, verbose=0)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss=[tf.keras.losses.MeanSquaredError(name='mse')],
                  metrics=[tf.keras.metrics.MeanSquaredError(name='mse')])

    x_tr, y_tr, x_val, y_val = data

    history = model.fit(x_tr, y_tr, validation_data=(x_val, y_val),
                        batch_size=batch_size, verbose = 1,
                        callbacks=[ReportCheckpointCallback(metrics=['val_mse']), es])
    hist = pd.DataFrame(history.history)
    score = hist['val_mse'].min()
    return {"score" : score}


if __name__ == '__main__':
    df = pd.read_feather('../data/df_btc_with_features_5m_spot.feather')

    df['target'] = df['close'].copy()

    start_time = df['open_time'].min()
    end_time = df['open_time'].max()
    dates = df['open_time'].unique()
    n = len(dates)
    train_idx = int(0.7 * n)
    test_idx = int(0.8 * n)

    train_df = df.iloc[:train_idx]
    valid_df = df.iloc[train_idx:test_idx]
    test_df = df.iloc[test_idx:]

    config = {
        "num_generic_neurons": tune.choice([200, 400, 500, 600, 700]),
        'num_generic_stacks': tune.choice([30, 40, 50, 60]),
        'num_generic_layers': tune.randint(3, 8),
        'num_trend_neurons': tune.choice([128, 256, 512]),
        'num_trend_stacks': tune.randint(3, 7),
        'num_trend_layers': tune.randint(3, 8),
        'polynomial_term': tune.randint(1, 5),
    }


    scaler = MinMaxScaler((0, 1))
    train_df['target'] = scaler.fit_transform(train_df[['close']])
    valid_df['target'] = scaler.transform(valid_df[['close']])

    hyperband = HyperBandScheduler(metric="score", mode="min")

    lookback = 200
    horizon = 100
    train_df = train_df.iloc[:500]
    valid_df = valid_df.iloc[:100]
    x_tr, y_tr = prep_time_series(train_df['target'], lookback = lookback, horizon = horizon)
    x_val, y_val = prep_time_series(valid_df['target'], lookback = lookback, horizon = horizon)

    algo = BayesOptSearch(random_search_steps=4)

    tuner = Tuner(
        tune.with_parameters(train_func, data = (x_tr, y_tr, x_val, y_val)),
        param_space = config,
        tune_config = tune.TuneConfig(
            metric = "score", mode = "min", search_alg = algo
        ),
        run_config = train.RunConfig(stop={"training_iteration": 3})
    )

    results = tuner.fit()