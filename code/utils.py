import pandas as pd
import numpy as np
from ta.momentum import (
    AwesomeOscillatorIndicator,
    KAMAIndicator,
    PercentagePriceOscillator,
    PercentageVolumeOscillator,
    ROCIndicator,
    RSIIndicator,
    StochasticOscillator,
    StochRSIIndicator,
    TSIIndicator,
    UltimateOscillator,
    WilliamsRIndicator,
)
from ta.others import (
    CumulativeReturnIndicator,
    DailyLogReturnIndicator,
    DailyReturnIndicator,
)
from ta.trend import (
    MACD,
    ADXIndicator,
    AroonIndicator,
    CCIIndicator,
    DPOIndicator,
    EMAIndicator,
    IchimokuIndicator,
    KSTIndicator,
    MassIndex,
    PSARIndicator,
    SMAIndicator,
    STCIndicator,
    TRIXIndicator,
    VortexIndicator,
)
from ta.volatility import (
    AverageTrueRange,
    BollingerBands,
    DonchianChannel,
    KeltnerChannel,
    UlcerIndex,
)
from ta.volume import (
    AccDistIndexIndicator,
    ChaikinMoneyFlowIndicator,
    EaseOfMovementIndicator,
    ForceIndexIndicator,
    MFIIndicator,
    NegativeVolumeIndexIndicator,
    OnBalanceVolumeIndicator,
    VolumePriceTrendIndicator,
    VolumeWeightedAveragePrice,
)

def add_momentum_ta(
    df: pd.DataFrame,
    high: str,
    low: str,
    close: str,
    volume: str,
    fillna: bool = False,
    colprefix: str = "",
    vectorized: bool = False,
) -> pd.DataFrame:
    """Add trend technical analysis features to dataframe.

    Args:
        df (pandas.core.frame.DataFrame): Dataframe base.
        high (str): Name of 'high' column.
        low (str): Name of 'low' column.
        close (str): Name of 'close' column.
        volume (str): Name of 'volume' column.
        fillna(bool): if True, fill nan values.
        colprefix(str): Prefix column names inserted
        vectorized(bool): if True, use only vectorized functions indicators

    Returns:
        pandas.core.frame.DataFrame: Dataframe with new features.
    """

    # Relative Strength Index (RSI)
    df[f"{colprefix}momentum_rsi"] = RSIIndicator(
        close=df[close], window=14, fillna=fillna
    ).rsi()

    # Stoch RSI (StochRSI)
    indicator_srsi = StochRSIIndicator(
        close=df[close], window=14, smooth1=3, smooth2=3, fillna=fillna
    )
    df[f"{colprefix}momentum_stoch_rsi"] = indicator_srsi.stochrsi()
    df[f"{colprefix}momentum_stoch_rsi_k"] = indicator_srsi.stochrsi_k()
    df[f"{colprefix}momentum_stoch_rsi_d"] = indicator_srsi.stochrsi_d()

    # TSI Indicator
    df[f"{colprefix}momentum_tsi"] = TSIIndicator(
        close=df[close], window_slow=25, window_fast=13, fillna=fillna
    ).tsi()

    # Ultimate Oscillator
    df[f"{colprefix}momentum_uo"] = UltimateOscillator(
        high=df[high],
        low=df[low],
        close=df[close],
        window1=7,
        window2=14,
        window3=28,
        weight1=4.0,
        weight2=2.0,
        weight3=1.0,
        fillna=fillna,
    ).ultimate_oscillator()

    # Stoch Indicator
    indicator_so = StochasticOscillator(
        high=df[high],
        low=df[low],
        close=df[close],
        window=14,
        smooth_window=3,
        fillna=fillna,
    )
    df[f"{colprefix}momentum_stoch"] = indicator_so.stoch()
    df[f"{colprefix}momentum_stoch_signal"] = indicator_so.stoch_signal()

    # Williams R Indicator
    df[f"{colprefix}momentum_wr"] = WilliamsRIndicator(
        high=df[high], low=df[low], close=df[close], lbp=14, fillna=fillna
    ).williams_r()

    # Awesome Oscillator
    df[f"{colprefix}momentum_ao"] = AwesomeOscillatorIndicator(
        high=df[high], low=df[low], window1=5, window2=34, fillna=fillna
    ).awesome_oscillator()

    # Rate Of Change
    df[f"{colprefix}momentum_roc"] = ROCIndicator(
        close=df[close], window=12, fillna=fillna
    ).roc()

    # Percentage Price Oscillator
    indicator_ppo = PercentagePriceOscillator(
        close=df[close], window_slow=26, window_fast=12, window_sign=9, fillna=fillna
    )
    df[f"{colprefix}momentum_ppo"] = indicator_ppo.ppo()
    df[f"{colprefix}momentum_ppo_signal"] = indicator_ppo.ppo_signal()
    df[f"{colprefix}momentum_ppo_hist"] = indicator_ppo.ppo_hist()

    # Percentage Volume Oscillator
    indicator_pvo = PercentageVolumeOscillator(
        volume=df[volume], window_slow=26, window_fast=12, window_sign=9, fillna=fillna
    )
    df[f"{colprefix}momentum_pvo"] = indicator_pvo.pvo()
    df[f"{colprefix}momentum_pvo_signal"] = indicator_pvo.pvo_signal()
    df[f"{colprefix}momentum_pvo_hist"] = indicator_pvo.pvo_hist()

    if not vectorized:
        # KAMA
        df[f"{colprefix}momentum_kama"] = KAMAIndicator(
            close=df[close], window=10, pow1=2, pow2=30, fillna=fillna
        ).kama()

    return df


def add_volume_ta(
    df: pd.DataFrame,
    high: str,
    low: str,
    close: str,
    volume: str,
    fillna: bool = False,
    colprefix: str = "",
    vectorized: bool = False,
) -> pd.DataFrame:
    """Add volume technical analysis features to dataframe.

    Args:
        df (pandas.core.frame.DataFrame): Dataframe base.
        high (str): Name of 'high' column.
        low (str): Name of 'low' column.
        close (str): Name of 'close' column.
        volume (str): Name of 'volume' column.
        fillna(bool): if True, fill nan values.
        colprefix(str): Prefix column names inserted
        vectorized(bool): if True, use only vectorized functions indicators

    Returns:
        pandas.core.frame.DataFrame: Dataframe with new features.
    """

    # Accumulation Distribution Index
    df[f"{colprefix}volume_adi"] = AccDistIndexIndicator(
        high=df[high], low=df[low], close=df[close], volume=df[volume], fillna=fillna
    ).acc_dist_index()

    # On Balance Volume
    df[f"{colprefix}volume_obv"] = OnBalanceVolumeIndicator(
        close=df[close], volume=df[volume], fillna=fillna
    ).on_balance_volume()

    # Chaikin Money Flow
    df[f"{colprefix}volume_cmf"] = ChaikinMoneyFlowIndicator(
        high=df[high], low=df[low], close=df[close], volume=df[volume], fillna=fillna
    ).chaikin_money_flow()

    # Force Index
    df[f"{colprefix}volume_fi"] = ForceIndexIndicator(
        close=df[close], volume=df[volume], window=13, fillna=fillna
    ).force_index()

    # Ease of Movement
    indicator_eom = EaseOfMovementIndicator(
        high=df[high], low=df[low], volume=df[volume], window=14, fillna=fillna
    )
    df[f"{colprefix}volume_em"] = indicator_eom.ease_of_movement()
    df[f"{colprefix}volume_sma_em"] = indicator_eom.sma_ease_of_movement()

    # Volume Price Trend
    df[f"{colprefix}volume_vpt"] = VolumePriceTrendIndicator(
        close=df[close], volume=df[volume], fillna=fillna
    ).volume_price_trend()

    # Volume Weighted Average Price
    df[f"{colprefix}volume_vwap"] = VolumeWeightedAveragePrice(
        high=df[high],
        low=df[low],
        close=df[close],
        volume=df[volume],
        window=14,
        fillna=fillna,
    ).volume_weighted_average_price()

    if not vectorized:
        # Money Flow Indicator
        df[f"{colprefix}volume_mfi"] = MFIIndicator(
            high=df[high],
            low=df[low],
            close=df[close],
            volume=df[volume],
            window=14,
            fillna=fillna,
        ).money_flow_index()

        # Negative Volume Index
        df[f"{colprefix}volume_nvi"] = NegativeVolumeIndexIndicator(
            close=df[close], volume=df[volume], fillna=fillna
        ).negative_volume_index()

    return df

# Two new features from the competition tutorial
def upper_shadow(df):
    return df["High"] - np.maximum(df["Close"], df["Open"])

def lower_shadow(df):
    return np.minimum(df["Close"], df["Open"]) - df["Low"]

