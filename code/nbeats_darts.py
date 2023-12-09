import pandas as pd
import gc
import numpy as np
from sklearn.preprocessing import MinMaxScaler


if __name__ == '__main__':
    df = pd.read_feather('../data/df_btc_with_features_5m_spot_new_features.feather')

    df['target'] = df['close'].pct_change(1)
    df = df.dropna(subset=['target'], axis=0)

    start_time = df['open_time'].min()
    end_time = df['open_time'].max()
    dates = df['open_time'].unique()
    n = len(dates)
    train_idx = int(0.7 * n)
    valid_idx = int(0.9 * n)
    train_end = dates[train_idx]
    valid_end = dates[valid_idx]

    train_df = df.loc[df['open_time'] < train_end].reset_index(drop=True)
    valid_df = df.loc[(train_end <= df['open_time']) & (df['open_time'] < valid_end)].reset_index(drop=True)

    train_df = pd.concat([train_df, valid_df], axis=0)

    test_df = df.loc[(df['open_time'] >= valid_end)].reset_index(drop=True)
    valid_df = test_df.copy()

    lookback = 200
    horizon = 100

    from pathlib import Path

    directory = 'nbeats_darts_test'
    lib = Path(f"../../output/{directory}").mkdir(parents=True, exist_ok=True)

    from darts.utils.callbacks import TFMProgressBar


    def generate_torch_kwargs():
        # run torch models on CPU, and disable progress bars for all model stages except training.
        return {
            "pl_trainer_kwargs": {
                "accelerator": "gpu",
                "callbacks": [TFMProgressBar(enable_train_bar_only=True)],
            }
        }


    from torchmetrics.regression import R2Score, MeanSquaredError
    from pytorch_lightning.callbacks.early_stopping import EarlyStopping
    import torch

    # es = EarlyStopping(monitor="val_loss", patience = 10, mode = "min")
    # ckp_path = f'../../output/{directory}/'
    # checkpoint_callback = ModelCheckpoint(dirpath='my/path',
    #                                       filename='{epoch}-{val_loss:.2f}-{other_metric:.2f}',
    #                                      mode = 'min')
    import tscv

    groups = pd.factorize(
        train_df['open_time'].dt.day.astype(str) + '_' + train_df['open_time'].dt.month.astype(str) + '_' + train_df[
            'open_time'].dt.year.astype(str))[0]

    cv = tscv.PurgedGroupTimeSeriesSplit(
        n_splits=5,
        group_gap=31,
    )

    config = {
        "input_chunk_length": lookback,
        "output_chunk_length": 100,
        "generic_architecture": True,
        "num_stacks": 20,
        "num_blocks": 3,
        "num_layers": 5,
        "layer_widths": 512,
        "expansion_coefficient_dim": 5,
        # "trend_polynomial_degree" : 3
        "dropout": 0.3,
        "activation": "ReLU",
        "torch_metrics": [R2Score(), MeanSquaredError()],
        "loss_fn": torch.nn.MSELoss(),
        "batch_size": 1024,
        "n_epochs": 100,
        "nr_epochs_val_period": 1,
        'save_checkpoints': True,
        **generate_torch_kwargs()
    }

    from sklearn.preprocessing import MinMaxScaler
    from darts.models import NBEATSModel

    model_name = "nbeats_run"

    for fold, (train_idx, val_idx) in enumerate(cv.split(train_df, train_df[f'target'], groups)):
        if fold >= 4:
            x_train, x_valid = train_df['target'].iloc[train_idx], train_df['target'].iloc[val_idx]

            min_train, max_train = min(train_df['open_time'].iloc[train_idx]).to_pydatetime(), max(
                train_df['open_time'].iloc[train_idx]).to_pydatetime()
            min_valid, max_valid = min(train_df['open_time'].iloc[val_idx]).to_pydatetime(), max(
                train_df['open_time'].iloc[val_idx]).to_pydatetime()

            scaler = MinMaxScaler()
            x_tr_scaled = scaler.fit_transform(x_train.values.reshape(-1, 1))
            x_val_scaled = scaler.transform(x_valid.values.reshape(-1, 1))

            model = NBEATSModel(**config)
            model.fit(series=pd.Series(x_tr_scaled.reshape(-1)), val_series=pd.Series(x_val_scaled.reshape(-1)))
