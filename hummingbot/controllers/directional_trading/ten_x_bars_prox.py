import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Union
from decimal import Decimal
from pydantic import Field

from hummingbot.client.config.config_data_types import ClientFieldData
from hummingbot.strategy_v2.models.executor_actions import CreateExecutorAction, StopExecutorAction
from controllers.directional_trading.smoothing import get_smoothing_function
from hummingbot.data_feed.candles_feed.candles_factory import CandlesFactory
from hummingbot.data_feed.candles_feed.data_types import CandlesConfig
from hummingbot.strategy_v2.controllers.directional_trading_controller_base import (
    DirectionalTradingControllerBase,
    DirectionalTradingControllerConfigBase,
)

class TenXBarsProXConfig(DirectionalTradingControllerConfigBase):
    controller_name: str = "ten_x_bars_prox"
    candles_config: List[CandlesConfig] = []
    connector: str
    trading_pair: str
    candle_interval: str
    bars_back: int
    order_amount: Decimal = Field(default=Decimal("0.1"), client_data=ClientFieldData(prompt=lambda msg: "Enter the order amount", prompt_on_new=True))
    dmi_length: int = Field(default=14, client_data=ClientFieldData(prompt=lambda msg: "Enter the DMI Length", prompt_on_new=True))
    adx_threshold: int = Field(default=20, client_data=ClientFieldData(prompt=lambda msg: "Enter the ADX Threshold", prompt_on_new=True))
    vol_avg_length: int = Field(default=20, client_data=ClientFieldData(prompt=lambda msg: "Enter the Volume Average Length", prompt_on_new=True))
    vol_per_threshold: int = Field(default=50, client_data=ClientFieldData(prompt=lambda msg: "Enter the Volume Percent Threshold", prompt_on_new=True))
    smooth_type: str = Field(default="Jurik Moving Average", client_data=ClientFieldData(prompt=lambda msg: "Enter the Smoothing Type", prompt_on_new=True))
    phase: float = Field(default=0, client_data=ClientFieldData(prompt=lambda msg: "Enter the Jurik Phase", prompt_on_new=True))
    power: float = Field(default=2, client_data=ClientFieldData(prompt=lambda msg: "Enter the Jurik Power", prompt_on_new=True))
    upper_deviation: float = Field(default=2.618, client_data=ClientFieldData(prompt=lambda msg: "Enter the Upper Deviation", prompt_on_new=True))
    lower_deviation: float = Field(default=2.618, client_data=ClientFieldData(prompt=lambda msg: "Enter the Lower Deviation", prompt_on_new=True))
    show_volume_strength_yellow: bool = Field(default=True, client_data=ClientFieldData(prompt=lambda msg: "Show Volume Strength on Yellow Bars", prompt_on_new=True))

def ten_x_bars_prox(
    df: pd.DataFrame,
    config: TenXBarsProXConfig
) -> Dict[str, pd.Series]:
    """
    Calculate 10x-Bars ProX signals.
    :param df: DataFrame with OHLCV data
    :param config: TenXBarsProXConfig object
    :return: Dictionary of signals
    """
    smoothing_func = get_smoothing_function(config.smooth_type)

    def apply_smoothing(series: pd.Series, length: int) -> pd.Series:
        if config.smooth_type == "Jurik Moving Average":
            return smoothing_func(series, length, phase=config.phase, power=config.power)
        return smoothing_func(series, length)

    # Calculate DMI and ADX
    up_move = df['high'].diff()
    down_move = -df['low'].diff()
    true_range = apply_smoothing(pd.concat([df['high'] - df['low'], 
                                            abs(df['high'] - df['close'].shift(1)), 
                                            abs(df['low'] - df['close'].shift(1))], axis=1).max(axis=1), 
                                 config.dmi_length)

    di_plus = 100 * apply_smoothing(np.where((up_move > down_move) & (up_move > 0), up_move, 0), config.dmi_length) / true_range
    di_minus = 100 * apply_smoothing(np.where((down_move > up_move) & (down_move > 0), down_move, 0), config.dmi_length) / true_range

    dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = apply_smoothing(dx, config.dmi_length)

    long = (di_plus > di_minus) & (adx > config.adx_threshold)
    short = (di_minus > di_plus) & (adx > config.adx_threshold)

    # Calculate volume strength
    avg_volume = apply_smoothing(df['volume'], config.vol_avg_length)
    volume_percent = 100 * (df['volume'] - avg_volume.shift(1)) / df['volume']
    volume_strength = (volume_percent >= config.vol_per_threshold) & (long | short | config.show_volume_strength_yellow)

    # Calculate linear regression channels
    def calc_slope(source: pd.Series, length: int) -> Tuple[float, float, float]:
        x = np.arange(length)
        y = source.iloc[-length:]
        slope, intercept = np.polyfit(x, y, 1)
        average = y.mean()
        return slope, average, intercept

    def calc_dev(source: pd.Series, length: int, slope: float, average: float, intercept: float) -> Tuple[float, float, float, float]:
        x = np.arange(length)
        y = source.iloc[-length:]
        y_pred = slope * x + intercept
        
        up_dev = (y - y_pred).clip(lower=0).max()
        down_dev = (y_pred - y).clip(lower=0).max()
        std_dev = np.std(y - y_pred)
        pearson_r = np.corrcoef(y, y_pred)[0, 1]
        
        return std_dev, pearson_r, up_dev, down_dev

    def calc_dev(source: pd.Series, length: int, slope: float, average: float, intercept: float) -> Tuple[float, float, float, float]:
        x = np.arange(length)
        y = source.iloc[-length:]
        y_pred = slope * x + intercept
        
        up_dev = (y - y_pred).clip(lower=0).max()
        down_dev = (y_pred - y).clip(lower=0).max()
        std_dev = np.std(y - y_pred)
        pearson_r = np.corrcoef(y, y_pred)[0, 1]
        
        return std_dev, pearson_r, up_dev, down_dev

    window = len(df)
    slope, average, intercept = calc_slope(df['close'], window)
    std_dev, pearson_r, up_dev, down_dev = calc_dev(df['close'], window, slope, average, intercept)

    start_price = intercept + slope * (window - 1)
    end_price = intercept

    upper_start_price = start_price + config.upper_deviation * std_dev
    upper_end_price = end_price + config.upper_deviation * std_dev
    lower_start_price = start_price - config.lower_deviation * std_dev
    lower_end_price = end_price - config.lower_deviation * std_dev

    upper_channel = pd.Series(np.linspace(upper_start_price, upper_end_price, window), index=df.index)
    lower_channel = pd.Series(np.linspace(lower_start_price, lower_end_price, window), index=df.index)
    base_line = pd.Series(np.linspace(start_price, end_price, window), index=df.index)

    # Calculate break above and break below signals
    break_above = (df['close'] > upper_channel) & (~df.index.isin([df.index[-1]]) | (df.index.isin([df.index[-1]]) & df['close'].notna()))
    break_below = (df['close'] < lower_channel) & (~df.index.isin([df.index[-1]]) | (df.index.isin([df.index[-1]]) & df['close'].notna()))

    return {
        'long': long,
        'short': short,
        'volume_strength': volume_strength,
        'upper_channel': upper_channel,
        'lower_channel': lower_channel,
        'base_line': base_line,
        'slope': pd.Series([slope] * len(df), index=df.index),
        'break_above': break_above,
        'break_below': break_below
    }

class TenXBarsProXController(DirectionalTradingControllerBase):
    def __init__(self, config: TenXBarsProXConfig, *args, **kwargs):
        super().__init__(config)
        self.config = config
        self.df = pd.DataFrame()
        self.max_records = config.dmi_length
        if len(self.candles_config) == 0:
            self.candles_config = [CandlesConfig(
                connector=config.connector,
                trading_pair=config.trading_pair,
                interval=config.candle_interval,
                max_records=config.bars_back
            )]
        self.candles_factory = CandlesFactory(self.candles_config)
        self.consecutive_long = 0
        self.consecutive_short = 0
        super().__init__(config, *args, **kwargs)

    async def process_tick(self) -> List[Union[CreateExecutorAction, StopExecutorAction]]:
        await self.update_market_data()
        signals = self.calculate_signals()
        return await self.process_signals(signals)

    async def update_market_data(self):
        candles = await self.candles_factory.get_candles(
            connector=self.config.connector,
            trading_pair=self.config.trading_pair,
            interval=self.config.candle_interval,
            max_records=self.config.bars_back
        )
        self.df = candles.candles_df

    def calculate_signals(self) -> Dict[str, pd.Series]:
        return ten_x_bars_prox(self.df, self.config)

    async def process_signals(self, signals: Dict[str, pd.Series]) -> List[Union[CreateExecutorAction, StopExecutorAction]]:
        actions = []

        if signals['long'].iloc[-1] and signals['volume_strength'].iloc[-1]:
            self.consecutive_long += 1
            self.consecutive_short = 0
            weight = 1 + min(self.consecutive_long * 0.1, 0.5)  # Max 50% increase
            actions.append(CreateExecutorAction(executor_type="buy", amount=self.config.order_amount * Decimal(str(weight))))
        elif signals['short'].iloc[-1] and signals['volume_strength'].iloc[-1]:
            self.consecutive_short += 1
            self.consecutive_long = 0
            weight = 1 + min(self.consecutive_short * 0.1, 0.5)  # Max 50% increase
            actions.append(CreateExecutorAction(executor_type="sell", amount=self.config.order_amount * Decimal(str(weight))))
        else:
            self.consecutive_long = 0
            self.consecutive_short = 0

        # Add actions for channel breaks
        if signals['break_above'].iloc[-1]:
            actions.append(CreateExecutorAction(executor_type="buy", amount=self.config.order_amount))
        elif signals['break_below'].iloc[-1]:
            actions.append(CreateExecutorAction(executor_type="sell", amount=self.config.order_amount))

        return actions

    def to_format_status(self) -> List[str]:
        signals = self.calculate_signals()
        last_values = {k: v.iloc[-1] for k, v in signals.items()}
        
        return [
            f"DMI Length: {self.config.dmi_length}",
            f"ADX Threshold: {self.config.adx_threshold}",
            f"Smoothing type: {self.config.smooth_type}",
            f"Upper Deviation: {self.config.upper_deviation}",
            f"Lower Deviation: {self.config.lower_deviation}",
            f"Long Signal: {'Yes' if last_values['long'] else 'No'}",
            f"Short Signal: {'Yes' if last_values['short'] else 'No'}",
            f"Volume Strength: {'Yes' if last_values['volume_strength'] else 'No'}",
            f"Break Above: {'Yes' if last_values['break_above'] else 'No'}",
            f"Break Below: {'Yes' if last_values['break_below'] else 'No'}",
            f"Slope: {last_values['slope']:.4f}",
        ]


# class TenXBarsProXStrategy(GenericV2StrategyWithCashOut):
#     def __init__(self, config: Dict):
#         super().__init__(config)
#         self.ten_x_bars_prox_controller = TenXBarsProXController(config["ten_x_bars_prox"])

#     async def create_actions_proposal(self) -> List[CreateExecutorAction]:
#         return await self.ten_x_bars_prox_controller.process_tick()

#     async def stop_actions_proposal(self) -> List[StopExecutorAction]:
#         # Implement logic to stop executors if needed
#         return []