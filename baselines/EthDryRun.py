# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401

import pandas as pd
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.strategy import IStrategy, IntParameter

class EthDryRun(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = '5m'

    # Baked in from your previous Hyperopt
    minimal_roi = {
        "0": 0.148,
        "22": 0.072,
        "49": 0.026,
        "60": 0
    }
    stoploss = -0.204
    
    # Let hyperopt take control of trailing stops
    trailing_stop = False
    trailing_stop_positive = None
    trailing_stop_positive_offset = 0.0
    trailing_only_offset_is_reached = False

    protections = [
        {"method": "CooldownPeriod", "stop_duration_candles": 5},
        {"method": "StoplossGuard", "lookback_period_candles": 24, "trade_limit": 4, "stop_duration_candles": 12, "only_per_pair": False}
    ]

    # Baked in defaults from your previous Hyperopt
    buy_rsi = IntParameter(15, 45, default=16, space='buy', optimize=True)
    sell_rsi = IntParameter(55, 85, default=60, space='sell', optimize=True)

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe['rsi'] = ta.RSI(dataframe)
        
        # --- NEW: Calculate the 200 EMA for the Trend Filter ---
        dataframe['ema_200'] = ta.EMA(dataframe, timeperiod=200)
        
        return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe.loc[
            (
                (qtpylib.crossed_below(dataframe['rsi'], self.buy_rsi.value)) &
                (dataframe['close'] > dataframe['ema_200']) & # <-- NEW: Only buy if price is above the 200 EMA
                (dataframe['volume'] > 0)
            ),
            ['enter_long', 'enter_tag']] = (1, 'rsi_buy')
        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe.loc[
            (
                (qtpylib.crossed_above(dataframe['rsi'], self.sell_rsi.value)) &
                (dataframe['volume'] > 0)
            ),
            ['exit_long', 'exit_tag']] = (1, 'rsi_sell')
        return dataframe