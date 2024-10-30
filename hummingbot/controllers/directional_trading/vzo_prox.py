import pandas as pd
import numpy as np
from typing import Tuple, List, Union
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

class VZOProXConfig(DirectionalTradingControllerConfigBase):
    controller_name: str = "vzo_prox"
    candles_config: List[CandlesConfig] = []
    connector: str
    trading_pair: str
    candle_interval: str
    bars_back: int
    vzo_length: int = Field(default=14, client_data=ClientFieldData(prompt=lambda msg: "Enter the VZO Length"))
    minor_sell_value: int = Field(default=40, client_data=ClientFieldData(prompt=lambda msg: "Enter the Minor Sell Value"))
    minor_buy_value: int = Field(default=-40, client_data=ClientFieldData(prompt=lambda msg: "Enter the Minor Buy Value"))
    minor_major_range: int = Field(default=20, client_data=ClientFieldData(prompt=lambda msg: "Enter the Minor Major Range"))
    zero_line_cross_filter_range: int = Field(default=20, client_data=ClientFieldData(prompt=lambda msg: "Enter the ZeroLine Cross Range"))
    vzo_noise: int = Field(default=2, client_data=ClientFieldData(prompt=lambda msg: "Enter the Noise Filter"))
    smooth_type: str = Field(default="Jurik Moving Average", client_data=ClientFieldData(prompt=lambda msg: "Enter the MA Type"))
    phase: float = Field(default=50, client_data=ClientFieldData(prompt=lambda msg: "Enter the Phase for JMA"))
    power: float = Field(default=2, client_data=ClientFieldData(prompt=lambda msg: "Enter the Power for JMA"))
    data_sample: int = Field(default=55, client_data=ClientFieldData(prompt=lambda msg: "Enter the Adaptive Filter Sample Length"))
    pcnt_above: float = Field(default=80, client_data=ClientFieldData(prompt=lambda msg: "Enter the Hi is Above X% of Sample"))
    pcnt_below: float = Field(default=80, client_data=ClientFieldData(prompt=lambda msg: "Enter the Lo is Below X% of Sample"))
    order_amount: Decimal = Field(default=Decimal("0.1"), client_data=ClientFieldData(prompt=lambda msg: "Enter the order amount"))
    trail_threshold: float = Field(default=0.986, client_data=ClientFieldData(prompt=lambda msg: "Enter the Trailing Signal Filter Threshold"))

def vzo_prox(
    df: pd.DataFrame,
    config: VZOProXConfig
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Calculate VZO-ProX signals.
    :param df: DataFrame with OHLCV data
    :param config: VZOProXConfig object
    :return: minor_buy, major_buy, minor_sell, major_sell, adaptive_z_buy, adaptive_z_sell, zero_line_cross_up, zero_line_cross_down, minor_sell_z, major_sell_z, minor_buy_z, major_buy_z, adaptive_z_cross_down, adaptive_z_cross_up signals
    """
    smoothing_func = get_smoothing_function(config.smooth_type)

    def apply_smoothing(series: pd.Series, length: int) -> pd.Series:
        if config.smooth_type == "Jurik Moving Average":
            return smoothing_func(series, length, phase=config.phase, power=config.power)
        return smoothing_func(series, length)

    vp = apply_smoothing(np.sign(df['close'] - df['close'].shift(1)) * df['volume'], config.vzo_length)
    tv = apply_smoothing(df['volume'], config.vzo_length)

    turbo_vzo = 100 * vp / tv
    turbo_vzo_e = apply_smoothing(turbo_vzo, config.vzo_noise)

    smpl_above = turbo_vzo.rolling(window=config.data_sample).quantile(config.pcnt_above / 100)
    smpl_below = turbo_vzo.rolling(window=config.data_sample).quantile(1 - config.pcnt_below / 100)

    minor_buy = pd.Series(np.where(
        (np.sign(turbo_vzo - turbo_vzo.shift(1)) > np.sign(turbo_vzo.shift(1) - turbo_vzo.shift(2))) &
        (turbo_vzo < config.minor_buy_value),
        turbo_vzo.shift(1),
        np.nan
    ))

    major_buy = pd.Series(np.where(
        (np.sign(turbo_vzo - turbo_vzo.shift(1)) > np.sign(turbo_vzo.shift(1) - turbo_vzo.shift(2))) &
        (turbo_vzo < config.minor_buy_value - config.minor_major_range),
        turbo_vzo.shift(1),
        np.nan
    ))

    minor_sell = pd.Series(np.where(
        (np.sign(turbo_vzo - turbo_vzo.shift(1)) < np.sign(turbo_vzo.shift(1) - turbo_vzo.shift(2))) &
        (turbo_vzo > config.minor_sell_value),
        turbo_vzo.shift(1),
        np.nan
    ))

    major_sell = pd.Series(np.where(
        (np.sign(turbo_vzo - turbo_vzo.shift(1)) < np.sign(turbo_vzo.shift(1) - turbo_vzo.shift(2))) &
        (turbo_vzo > config.minor_sell_value + config.minor_major_range),
        turbo_vzo.shift(1),
        np.nan
    ))

    adaptive_z_buy = pd.Series(np.where(
        (np.sign(turbo_vzo - turbo_vzo.shift(1)) > np.sign(turbo_vzo.shift(1) - turbo_vzo.shift(2))) &
        (turbo_vzo < smpl_below),
        turbo_vzo.shift(1),
        np.nan
    ))

    adaptive_z_sell = pd.Series(np.where(
        (np.sign(turbo_vzo - turbo_vzo.shift(1)) < np.sign(turbo_vzo.shift(1) - turbo_vzo.shift(2))) &
        (turbo_vzo > smpl_above),
        turbo_vzo.shift(1),
        np.nan
    ))

    zero_line_cross_down = pd.Series(np.where(
        (turbo_vzo <= 0) &
        (turbo_vzo.shift(1) > 0) &
        (turbo_vzo.shift(2) - turbo_vzo_e > config.zero_line_cross_filter_range),
        1,
        0
    ))

    zero_line_cross_up = pd.Series(np.where(
        (turbo_vzo >= 0) &
        (turbo_vzo.shift(1) < 0) &
        (turbo_vzo_e - turbo_vzo.shift(2) > config.zero_line_cross_filter_range),
        1,
        0
    ))

    minor_sell_z = pd.Series(np.where(
        (turbo_vzo >= config.minor_sell_value) &
        (turbo_vzo < config.minor_sell_value + config.minor_major_range) &
        (turbo_vzo.shift(1) < config.minor_sell_value),
        1,
        0
    ))

    major_sell_z = pd.Series(np.where(
        (turbo_vzo >= config.minor_sell_value + config.minor_major_range) &
        (turbo_vzo.shift(1) < config.minor_sell_value + config.minor_major_range),
        1,
        0
    ))

    minor_buy_z = pd.Series(np.where(
        (turbo_vzo <= config.minor_buy_value) &
        (turbo_vzo > config.minor_buy_value - config.minor_major_range) &
        (turbo_vzo.shift(1) > config.minor_buy_value),
        1,
        0
    ))

    major_buy_z = pd.Series(np.where(
        (turbo_vzo <= config.minor_buy_value - config.minor_major_range) &
        (turbo_vzo.shift(1) > config.minor_buy_value - config.minor_major_range),
        1,
        0
    ))

    adaptive_z_cross_down = pd.Series(np.where(
        (turbo_vzo <= smpl_above) &
        (turbo_vzo.shift(1) > smpl_above) &
        (turbo_vzo.shift(2) - turbo_vzo_e > config.zero_line_cross_filter_range),
        1,
        0
    ))

    adaptive_z_cross_up = pd.Series(np.where(
        (turbo_vzo >= smpl_below) &
        (turbo_vzo.shift(1) < smpl_below) &
        (turbo_vzo_e - turbo_vzo.shift(2) > config.zero_line_cross_filter_range),
        1,
        0
    ))

    return minor_buy, major_buy, minor_sell, major_sell, adaptive_z_buy, adaptive_z_sell, zero_line_cross_up, zero_line_cross_down, minor_sell_z, major_sell_z, minor_buy_z, major_buy_z, adaptive_z_cross_down, adaptive_z_cross_up

class VZOProXController(DirectionalTradingControllerBase):
    def __init__(self, config: VZOProXConfig, *args, **kwargs):
        super().__init__(config)
        self.config = config
        self.df = pd.DataFrame()
        if len(self.candles_config) == 0:
            self.candles_config = [CandlesConfig(
                connector=config.connector,
                trading_pair=config.trading_pair,
                interval=config.candle_interval,
                max_records=config.bars_back
            )]
        self.candles_factory = CandlesFactory(self.candles_config)
        self.consecutive_buy = 0
        self.consecutive_sell = 0
        super().__init__(config, *args, **kwargs)

    async def update_market_data(self):
        candles = await self.candles_factory.get_candles(
            connector=self.config.connector,
            trading_pair=self.config.trading_pair,
            interval=self.config.candle_interval,
            max_records=self.config.bars_back
        )
        self.df = candles.candles_df

    def calculate_signals(self):
        return vzo_prox(self.df, self.config)

    async def process_signals(self, signals) -> List[Union[CreateExecutorAction, StopExecutorAction]]:
        minor_buy, major_buy, minor_sell, major_sell, adaptive_z_buy, adaptive_z_sell, zero_line_cross_up, zero_line_cross_down, minor_sell_z, major_sell_z, minor_buy_z, major_buy_z, adaptive_z_cross_down, adaptive_z_cross_up = signals
        actions = []

        if major_buy.iloc[-1] or (minor_buy.iloc[-1] and zero_line_cross_up.iloc[-1]) or (adaptive_z_buy.iloc[-1] and adaptive_z_cross_up.iloc[-1]):
            self.consecutive_buy += 1
            self.consecutive_sell = 0
            weight = 1 + min(self.consecutive_buy * 0.1, 0.5)  # Max 50% increase
            actions.append(CreateExecutorAction(executor_type="buy", amount=self.config.order_amount * Decimal(str(weight))))
        elif major_sell.iloc[-1] or (minor_sell.iloc[-1] and zero_line_cross_down.iloc[-1]) or (adaptive_z_sell.iloc[-1] and adaptive_z_cross_down.iloc[-1]):
            self.consecutive_sell += 1
            self.consecutive_buy = 0
            weight = 1 + min(self.consecutive_sell * 0.1, 0.5)  # Max 50% increase
            actions.append(CreateExecutorAction(executor_type="sell", amount=self.config.order_amount * Decimal(str(weight))))
        else:
            self.consecutive_buy = 0
            self.consecutive_sell = 0

        return actions

    def to_format_status(self) -> List[str]:
        return [
            f"VZO Length: {self.config.vzo_length}",
            f"Smoothing type: {self.config.smooth_type}",
            f"Minor Buy Value: {self.config.minor_buy_value}",
            f"Minor Sell Value: {self.config.minor_sell_value}",
            f"Minor Major Range: {self.config.minor_major_range}",
        ]


# class VZOProXStrategy(GenericV2StrategyWithCashOut):
#     def __init__(self, config: Dict):
#         super().__init__(config)
#         self.vzo_prox_controller = VZOProXController(config["vzo_prox"])
#
#     async def create_actions_proposal(self) -> List[CreateExecutorAction]:
#         return await self.vzo_prox_controller.process_tick()
#
#     async def stop_actions_proposal(self) -> List[StopExecutorAction]:
#         # Implement logic to stop executors if needed
#         return []
