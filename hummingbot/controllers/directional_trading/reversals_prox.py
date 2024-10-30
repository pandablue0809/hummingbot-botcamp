import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Union
from decimal import Decimal
from pydantic import Field

from hummingbot.strategy_v2.controllers.controller_base import ControllerBase, ControllerConfigBase
from hummingbot.client.config.config_data_types import ClientFieldData
from hummingbot.strategy_v2.models.executor_actions import CreateExecutorAction, StopExecutorAction
from hummingbot.data_feed.candles_feed.candles_factory import CandlesFactory
from hummingbot.data_feed.candles_feed.data_types import CandlesConfig
from hummingbot.strategy_v2.controllers.directional_trading_controller_base import (
    DirectionalTradingControllerBase,
    DirectionalTradingControllerConfigBase,
)


class ReversalsProXConfig(DirectionalTradingControllerConfigBase):
    lookback_period: int = Field(default=20, client_data=ClientFieldData(prompt=lambda msg: "Enter the Lookback Period"))
    atr_multiplier: float = Field(default=2.0, client_data=ClientFieldData(prompt=lambda msg: "Enter the Bar ATR Multiplier"))
    bar_body_percent_min: float = Field(default=0.65, client_data=ClientFieldData(prompt=lambda msg: "Enter the Minimum Bar Body %"))
    bar_body_percent_max: float = Field(default=0.85, client_data=ClientFieldData(prompt=lambda msg: "Enter the Maximum Bar Body %"))
    o_lookback_period: int = Field(default=20, client_data=ClientFieldData(prompt=lambda msg: "Enter the Outside Bar Lookback Period"))
    o_atr_multiplier: float = Field(default=1.05, client_data=ClientFieldData(prompt=lambda msg: "Enter the Outside Bar ATR Multiplier"))
    connector: str = Field(default="binance", client_data=ClientFieldData(prompt=lambda msg: "Enter the connector name"))
    trading_pair: str = Field(default="BTC-USDT", client_data=ClientFieldData(prompt=lambda msg: "Enter the trading pair"))
    candle_interval: str = Field(default="1h", client_data=ClientFieldData(prompt=lambda msg: "Enter the candle interval"))
    order_amount: Decimal = Field(default=Decimal("0.01"), client_data=ClientFieldData(prompt=lambda msg: "Enter the order amount"))

def reversals_prox(df: pd.DataFrame, config: ReversalsProXConfig) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    typical_atr = df['high'].rolling(window=config.lookback_period).max() - df['low'].rolling(window=config.lookback_period).min()
    first_bar_body_size = abs(df['close'].shift(1) - df['open'].shift(1))
    first_bar_range = df['high'].shift(1) - df['low'].shift(1)
    first_bar_body_pct = first_bar_body_size / first_bar_range

    first_bar_signal = (first_bar_range > (config.atr_multiplier * typical_atr)) & \
                       (first_bar_body_pct >= config.bar_body_percent_min) & \
                       (first_bar_body_pct <= config.bar_body_percent_max)
    
    bull_signal = first_bar_signal & (df['close'].shift(1) < df['open'].shift(1)) & (df['close'] > df['open'])
    bear_signal = first_bar_signal & (df['close'].shift(1) > df['open'].shift(1)) & (df['close'] < df['open'])

    o_typical_atr = df['high'].rolling(window=config.o_lookback_period).max() - df['low'].rolling(window=config.o_lookback_period).min()
    o_first_bar_range = df['high'].shift(1) - df['low'].shift(1)

    o_first_bar_signal = (o_first_bar_range > (config.o_atr_multiplier * o_typical_atr))
    o_bull_signal = (df['low'] < df['low'].shift(1)) & (df['close'] > df['high'].shift(1)) & o_first_bar_signal
    o_bear_signal = (df['high'] > df['high'].shift(1)) & (df['close'] < df['low'].shift(1)) & o_first_bar_signal

    return bull_signal, bear_signal, o_bull_signal, o_bear_signal

class ReversalsProXController(DirectionalTradingControllerBase):
    def __init__(self, config: ReversalsProXConfig):
        super().__init__(config)
        self.config = config
        self.df = pd.DataFrame()
        self.candles_config = CandlesConfig(
            connector=config.connector,
            trading_pair=config.trading_pair,
            interval=config.candle_interval,
            max_records=config.lookback_period
        )
        self.candles_factory = CandlesFactory(self.candles_config)

    async def update_market_data(self):
        candles = await self.candles_factory.get_candles(
            connector=self.config.connector,
            trading_pair=self.config.trading_pair,
            interval=self.config.candle_interval,
            max_records=self.config.lookback_period
        )
        self.df = candles.candles_df

    def calculate_signals(self):
        bull_signal, bear_signal, o_bull_signal, o_bear_signal = reversals_prox(self.df, self.config)
        return {
            "bull_signal": bull_signal,
            "bear_signal": bear_signal,
            "o_bull_signal": o_bull_signal,
            "o_bear_signal": o_bear_signal
        }

    async def process_signals(self, signals) -> List[Union[CreateExecutorAction, StopExecutorAction]]:
        actions = []

        if signals["bull_signal"].iloc[-1] or signals["o_bull_signal"].iloc[-1]:
            actions.append(CreateExecutorAction(executor_type="buy", amount=self.config.order_amount))
        elif signals["bear_signal"].iloc[-1] or signals["o_bear_signal"].iloc[-1]:
            actions.append(CreateExecutorAction(executor_type="sell", amount=self.config.order_amount))

        return actions

    async def process_tick(self) -> List[Union[CreateExecutorAction, StopExecutorAction]]:
        await self.update_market_data()
        signals = self.calculate_signals()
        return await self.process_signals(signals)

    def to_format_status(self) -> List[str]:
        return [
            f"Lookback Period: {self.config.lookback_period}",
            f"ATR Multiplier: {self.config.atr_multiplier}",
            f"Bar Body % Range: {self.config.bar_body_percent_min} - {self.config.bar_body_percent_max}",
            f"Outside Bar Lookback Period: {self.config.o_lookback_period}",
            f"Outside Bar ATR Multiplier: {self.config.o_atr_multiplier}",
        ]
