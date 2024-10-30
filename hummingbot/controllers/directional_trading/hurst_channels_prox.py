import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Union, Set
from decimal import Decimal
from pydantic import Field

from hummingbot.client.config.config_data_types import ClientFieldData
from hummingbot.strategy_v2.models.executor_actions import CreateExecutorAction, StopExecutorAction
from hummingbot.data_feed.candles_feed.candles_factory import CandlesFactory
from hummingbot.data_feed.candles_feed.data_types import CandlesConfig
from .smoothing import get_smoothing_function
from hummingbot.strategy_v2.controllers.directional_trading_controller_base import (
    DirectionalTradingControllerBase,
    DirectionalTradingControllerConfigBase,
)

class HurstChannelsProXConfig(DirectionalTradingControllerConfigBase):
    controller_name: str = "hurst_channels_prox"
    candles_config: List[CandlesConfig] = []
    connector: str = Field(default="binance", client_data=ClientFieldData(prompt=lambda msg: "Enter the exchange connector", prompt_on_new=True))
    trading_pair: str = Field(default="BTC-USDT", client_data=ClientFieldData(prompt=lambda msg: "Enter the trading pair", prompt_on_new=True))
    candle_interval: str = Field(default="1h", client_data=ClientFieldData(prompt=lambda msg: "Enter the candle interval", prompt_on_new=True))
    bars_back: int = Field(default=500, client_data=ClientFieldData(prompt=lambda msg: "Enter the number of bars to look back", prompt_on_new=True))
    markets: Dict[str, Set[str]] = {}
    len1: int = Field(default=32, client_data=ClientFieldData(prompt=lambda msg: "Enter the length for the first channel", prompt_on_new=True))
    oct1: float = Field(default=100.0, client_data=ClientFieldData(prompt=lambda msg: "Enter the octave for the first channel", prompt_on_new=True))
    
    len2: int = Field(default=128, client_data=ClientFieldData(prompt=lambda msg: "Enter the length for the second channel", prompt_on_new=True))
    oct2: float = Field(default=200.0, client_data=ClientFieldData(prompt=lambda msg: "Enter the octave for the second channel", prompt_on_new=True))
    
    len3: int = Field(default=256, client_data=ClientFieldData(prompt=lambda msg: "Enter the length for the third channel", prompt_on_new=True))
    oct3: float = Field(default=400.0, client_data=ClientFieldData(prompt=lambda msg: "Enter the octave for the third channel", prompt_on_new=True))
    
    smooth_type: str = Field(default="Simple Moving Average", client_data=ClientFieldData(prompt=lambda msg: "Enter the smoothing type", prompt_on_new=True))
    phase: float = Field(default=0.0, client_data=ClientFieldData(prompt=lambda msg: "Enter the Jurik phase (if applicable)", prompt_on_new=True))
    power: float = Field(default=2.0, client_data=ClientFieldData(prompt=lambda msg: "Enter the Jurik power (if applicable)", prompt_on_new=True))
    
    order_amount: Decimal = Field(default=Decimal("0.1"), client_data=ClientFieldData(prompt=lambda msg: "Enter the order amount", prompt_on_new=True))
    max_repetitive_signals: int = Field(default=3, client_data=ClientFieldData(prompt=lambda msg: "Enter the maximum number of repetitive signals", prompt_on_new=True))

def hurst_channels_prox(df: pd.DataFrame, config: HurstChannelsProXConfig) -> Dict[str, pd.Series]:
    smoothing_func = get_smoothing_function(config.smooth_type)
    
    def calculate_channel(length: int, octave: float) -> Tuple[pd.Series, pd.Series, pd.Series]:
        if config.smooth_type == "Jurik Moving Average":
            ma = smoothing_func(df['ohlc4'], length, phase=config.phase, power=config.power)
        else:
            ma = smoothing_func(df['ohlc4'], length)
        
        upper = ma + octave
        lower = ma - octave
        
        return ma, upper, lower
    
    df['ohlc4'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    
    ma1, upper1, lower1 = calculate_channel(config.len1, config.oct1)
    ma2, upper2, lower2 = calculate_channel(config.len2, config.oct2)
    ma3, upper3, lower3 = calculate_channel(config.len3, config.oct3)
    
    # Calculate extended lines
    def calculate_extended_line(ma: pd.Series, length: int) -> pd.Series:
        slope = (ma - ma.shift(10)) * 1.6
        return ma + slope.shift(-int(length/2))
    
    extend1 = calculate_extended_line(ma1, config.len1)
    extend2 = calculate_extended_line(ma2, config.len2)
    extend3 = calculate_extended_line(ma3, config.len3)
    
    # Calculate weighted cross signals
    def calculate_weighted_cross_signals(price: pd.Series, ma: pd.Series, weight: float) -> Tuple[pd.Series, pd.Series]:
        upward_cross = ((price > ma) & (price.shift(1) <= ma.shift(1))) * weight
        downward_cross = ((price < ma) & (price.shift(1) >= ma.shift(1))) * weight
        return upward_cross, downward_cross
    
    upward_cross1, downward_cross1 = calculate_weighted_cross_signals(df['close'], extend1, 1.0)
    upward_cross2, downward_cross2 = calculate_weighted_cross_signals(df['close'], extend2, 2.0)
    upward_cross3, downward_cross3 = calculate_weighted_cross_signals(df['close'], extend3, 3.0)
    
    total_upward_cross = upward_cross1 + upward_cross2 + upward_cross3
    total_downward_cross = downward_cross1 + downward_cross2 + downward_cross3

    return {
        'ma1': ma1, 'upper1': upper1, 'lower1': lower1,
        'ma2': ma2, 'upper2': upper2, 'lower2': lower2,
        'ma3': ma3, 'upper3': upper3, 'lower3': lower3,
        'extend1': extend1,
        'extend1_upper': extend1 + config.oct1,
        'extend1_lower': extend1 - config.oct1,
        'extend2': extend2,
        'extend2_upper': extend2 + config.oct2,
        'extend2_lower': extend2 - config.oct2,
        'extend3': extend3,
        'extend3_upper': extend3 + config.oct3,
        'extend3_lower': extend3 - config.oct3,
        'upward_cross1': upward_cross1,
        'downward_cross1': downward_cross1,
        'upward_cross2': upward_cross2,
        'downward_cross2': downward_cross2,
        'upward_cross3': upward_cross3,
        'downward_cross3': downward_cross3,
        'total_upward_cross': total_upward_cross,
        'total_downward_cross': total_downward_cross,
    }

class HurstChannelsProXController(DirectionalTradingControllerBase):
    def __init__(self, config: HurstChannelsProXConfig):
        super().__init__(config)
        self.config = config
        self.df = pd.DataFrame()
        self.candles_config = CandlesConfig(
            connector=config.connector,
            trading_pair=config.trading_pair,
            interval=config.candle_interval,
            max_records=config.bars_back
        )
        self.candles_factory = CandlesFactory(self.candles_config)
        self.consecutive_buy = 0
        self.consecutive_sell = 0

    async def update_market_data(self):
        candles = await self.candles_factory.get_candles(
            connector=self.config.connector,
            trading_pair=self.config.trading_pair,
            interval=self.config.candle_interval,
            max_records=self.config.bars_back
        )
        self.df = candles.candles_df

    async def process_tick(self) -> List[Union[CreateExecutorAction, StopExecutorAction]]:
        await self.update_market_data()
        channels = self.calculate_channels()
        return self.generate_signals(channels)

    def calculate_channels(self) -> Dict[str, pd.Series]:
        return hurst_channels_prox(self.df, self.config)

    def generate_signals(self, channels: Dict[str, pd.Series]) -> List[Union[CreateExecutorAction, StopExecutorAction]]:
        actions = []
        
        total_upward_cross = channels['total_upward_cross'].iloc[-1]
        total_downward_cross = channels['total_downward_cross'].iloc[-1]
        
        if total_upward_cross > 0:
            self.consecutive_buy += 1
            self.consecutive_sell = 0
            if self.consecutive_buy <= self.config.max_repetitive_signals:
                weight = 1 + min(self.consecutive_buy * 0.1, 0.5)  # Max 50% increase
                actions.append(CreateExecutorAction(executor_type="buy", amount=self.config.order_amount * Decimal(str(weight))))
        elif total_downward_cross > 0:
            self.consecutive_sell += 1
            self.consecutive_buy = 0
            if self.consecutive_sell <= self.config.max_repetitive_signals:
                weight = 1 + min(self.consecutive_sell * 0.1, 0.5)  # Max 50% increase
                actions.append(CreateExecutorAction(executor_type="sell", amount=self.config.order_amount * Decimal(str(weight))))
        else:
            self.consecutive_buy = 0
            self.consecutive_sell = 0
        
        return actions

    def to_format_status(self) -> List[str]:
        channels = self.calculate_channels()
        last_values = {k: v.iloc[-1] for k, v in channels.items()}
        
        status = [
            f"Smoothing Type: {self.config.smooth_type}",
            f"Channel 1 (len={self.config.len1}, oct={self.config.oct1}):",
            f"  MA: {last_values['ma1']:.2f}",
            f"  Upper: {last_values['upper1']:.2f}",
            f"  Lower: {last_values['lower1']:.2f}",
            f"  Extended MA: {last_values['extend1']:.2f}",
            f"Channel 2 (len={self.config.len2}, oct={self.config.oct2}):",
            f"  MA: {last_values['ma2']:.2f}",
            f"  Upper: {last_values['upper2']:.2f}",
            f"  Lower: {last_values['lower2']:.2f}",
            f"  Extended MA: {last_values['extend2']:.2f}",
            f"Channel 3 (len={self.config.len3}, oct={self.config.oct3}):",
            f"  MA: {last_values['ma3']:.2f}",
            f"  Upper: {last_values['upper3']:.2f}",
            f"  Lower: {last_values['lower3']:.2f}",
            f"  Extended MA: {last_values['extend3']:.2f}",
            f"Total Upward Cross: {last_values['total_upward_cross']:.2f}",
            f"Total Downward Cross: {last_values['total_downward_cross']:.2f}",
            f"Consecutive Buy Signals: {self.consecutive_buy}",
            f"Consecutive Sell Signals: {self.consecutive_sell}",
        ]
        
        return status

# class HurstChannelsProXStrategy(GenericV2StrategyWithCashOut):
#     def __init__(self, config: Dict):
#         super().__init__(config)
#         self.hurst_channels_prox_controller = HurstChannelsProXController(config["hurst_channels_prox"])
#
#     async def create_actions_proposal(self) -> List[CreateExecutorAction]:
#         return await self.hurst_channels_prox_controller.process_tick()
#
#     async def stop_actions_proposal(self) -> List[StopExecutorAction]:
#         return []
