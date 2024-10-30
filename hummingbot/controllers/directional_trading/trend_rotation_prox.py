import pandas as pd
import numpy as np
from typing import Dict, List, Union
from decimal import Decimal
from pydantic import Field

from hummingbot.strategy_v2.controllers.controller_base import ControllerBase, ControllerConfigBase
from hummingbot.client.config.config_data_types import ClientFieldData
from hummingbot.strategy_v2.models.executor_actions import CreateExecutorAction, StopExecutorAction
from hummingbot.data_feed.candles_feed.candles_factory import CandlesFactory
from hummingbot.data_feed.candles_feed.data_types import CandlesConfig
from .smoothing import available_smoothing_functions, get_smoothing_function
from hummingbot.strategy_v2.controllers.directional_trading_controller_base import (
    DirectionalTradingControllerBase,
    DirectionalTradingControllerConfigBase,
)

class TrendRotationProXConfig(DirectionalTradingControllerConfigBase):
    connector: str
    trading_pair: str
    candle_interval: str
    bars_back: int
    fast_length: int = Field(default=3, client_data=ClientFieldData(prompt=lambda msg: "Enter the fast length"))
    slow_length: int = Field(default=10, client_data=ClientFieldData(prompt=lambda msg: "Enter the slow length"))
    turn_threshold: float = Field(default=0.0, client_data=ClientFieldData(prompt=lambda msg: "Enter the turn threshold"))
    number_of_timeframes: int = Field(default=6, client_data=ClientFieldData(prompt=lambda msg: "Enter the number of timeframes"))
    higher_timeframes: List[str] = Field(default=["10m", "5m", "4m", "3m", "2m", "1m"], client_data=ClientFieldData(prompt=lambda msg: "Enter the higher timeframes"))
    order_amount: Decimal = Field(default=Decimal("0.1"), client_data=ClientFieldData(prompt="Enter the order amount"))
    color_price: str = Field(default="aggregate", client_data=ClientFieldData(prompt=lambda msg: "Enter color price option (none, aggregate, HTF0, HTF1, HTF2, HTF3, HTF4, HTF5)"))
    alternate_oscillator_colors: bool = Field(default=False, client_data=ClientFieldData(prompt=lambda msg: "Use alternate oscillator colors?"))
    fast_ma_type: str = Field(
        default="ema",
        client_data=ClientFieldData(
            prompt=lambda msg: f"Enter fast MA type ({', '.join(available_smoothing_functions)})"
        )
    )
    slow_ma_type: str = Field(
        default="ema",
        client_data=ClientFieldData(
            prompt=lambda msg: f"Enter slow MA type ({', '.join(available_smoothing_functions)})"
        )
    )

def calculate_trend_rotation(df: pd.DataFrame, config: TrendRotationProXConfig) -> Dict[str, pd.Series]:
    def calculate_ma_diff(data: pd.DataFrame, fast_length: int, slow_length: int, fast_ma_type: str, slow_ma_type: str) -> pd.Series:
        fast_ma_func = get_smoothing_function(fast_ma_type)
        slow_ma_func = get_smoothing_function(slow_ma_type)
        fast_ma = fast_ma_func(data['close'], fast_length)
        slow_ma = slow_ma_func(data['close'], slow_length)
        return fast_ma - slow_ma

    def calculate_cross(value: pd.Series) -> pd.Series:
        cross_up = (value > 0) & (value.shift(1) <= 0)
        cross_down = (value < 0) & (value.shift(1) >= 0)
        return pd.Series(np.where(cross_up, 1, np.where(cross_down, -1, 0)))

    def calculate_turn(value: pd.Series, turn_threshold: float) -> pd.Series:
        turn_up = (value > value.shift(1)) & (value.shift(2) >= value.shift(1)) & ((value.shift(1) < -turn_threshold) | (value.shift(1) > 0))
        turn_down = (value < value.shift(1)) & (value.shift(2) <= value.shift(1)) & ((value.shift(1) > turn_threshold) | (value.shift(1) < 0))
        return pd.Series(np.where(turn_up, 1, np.where(turn_down, -1, 0)))

    signals = {}
    for i, timeframe in enumerate(config.higher_timeframes[:config.number_of_timeframes]):
        resampled_data = df.resample(timeframe).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last'
        })
        ma_diff = calculate_ma_diff(resampled_data, config.fast_length, config.slow_length, config.fast_ma_type, config.slow_ma_type)
        signals[f'cross_{i}'] = calculate_cross(ma_diff)
        signals[f'turn_{i}'] = calculate_turn(ma_diff, config.turn_threshold)
        signals[f'value_{i}'] = ma_diff

    return signals

class TrendRotationProXController(DirectionalTradingControllerBase):
    def __init__(self, config: TrendRotationProXConfig):
        super().__init__(config)
        self.config = config
        
        # Validate MA types
        if self.config.fast_ma_type not in available_smoothing_functions:
            raise ValueError(f"Invalid fast MA type. Choose from: {', '.join(available_smoothing_functions)}")
        if self.config.slow_ma_type not in available_smoothing_functions:
            raise ValueError(f"Invalid slow MA type. Choose from: {', '.join(available_smoothing_functions)}")
        
        self.df = pd.DataFrame()
        self.candles_config = CandlesConfig(
            connector=config.connector,
            trading_pair=config.trading_pair,
            interval=config.candle_interval,
            max_records=config.bars_back
        )
        self.candles_factory = CandlesFactory(self.candles_config)

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
        signals = calculate_trend_rotation(self.df, self.config)
        return await self.process_signals(signals)

    async def process_signals(self, signals: Dict[str, pd.Series]) -> List[Union[CreateExecutorAction, StopExecutorAction]]:
        actions = []
        cross_signals = [signals[f'cross_{i}'].iloc[-1] for i in range(self.config.number_of_timeframes)]
        turn_signals = [signals[f'turn_{i}'].iloc[-1] for i in range(self.config.number_of_timeframes)]
        value_signals = [signals[f'value_{i}'].iloc[-1] for i in range(self.config.number_of_timeframes)]

        sum_p = sum(1 for signal in cross_signals if signal == 1)
        sum_n = sum(1 for signal in cross_signals if signal == -1)
        sum_pt = sum(1 for signal in turn_signals if signal == 1)
        sum_nt = sum(1 for signal in turn_signals if signal == -1)

        if sum_p > 0 or sum_pt > 0:
            actions.append(CreateExecutorAction(executor_type="buy", amount=self.config.order_amount))
        elif sum_n > 0 or sum_nt > 0:
            actions.append(CreateExecutorAction(executor_type="sell", amount=self.config.order_amount))

        return actions

    def get_price_color(self, signals: Dict[str, pd.Series]) -> str:
        if self.config.color_price == "aggregate":
            sum_p = sum(1 for i in range(self.config.number_of_timeframes) if signals[f'value_{i}'].iloc[-1] > signals[f'value_{i}'].iloc[-2])
            sum_n = sum(1 for i in range(self.config.number_of_timeframes) if signals[f'value_{i}'].iloc[-1] < signals[f'value_{i}'].iloc[-2])
            if sum_p == 6:
                return "Price Bar Bullish 6 of 6"
            elif sum_p == 5:
                return "Price Bar Bullish 5 of 6"
            elif sum_p == 4:
                return "Price Bar Bullish 4 of 6"
            elif sum_n == 6:
                return "Price Bar Bearish 6 of 6"
            elif sum_n == 5:
                return "Price Bar Bearish 5 of 6"
            elif sum_n == 4:
                return "Price Bar Bearish 4 of 6"
            else:
                return "Price Bar Neutral"
        elif self.config.color_price.startswith("HTF"):
            htf_index = int(self.config.color_price[3:])
            if signals[f'value_{htf_index}'].iloc[-1] > signals[f'value_{htf_index}'].iloc[-2]:
                return "Rising Oscillator"
            else:
                return "Falling Oscillator"
        else:
            return "Current"

    def to_format_status(self) -> List[str]:
        return [
            f"Fast Length: {self.config.fast_length}",
            f"Slow Length: {self.config.slow_length}",
            f"Turn Threshold: {self.config.turn_threshold}",
            f"Number of Timeframes: {self.config.number_of_timeframes}",
            f"Higher Timeframes: {', '.join(self.config.higher_timeframes[:self.config.number_of_timeframes])}",
            f"Order Amount: {self.config.order_amount}",
            f"Color Price: {self.config.color_price}",
            f"Alternate Oscillator Colors: {self.config.alternate_oscillator_colors}",
            f"Fast MA Type: {self.config.fast_ma_type}",
            f"Slow MA Type: {self.config.slow_ma_type}",
        ]

# class TrendRotationProXStrategy(GenericV2StrategyWithCashOut):
#     def __init__(self, config: Dict):
#         super().__init__(config)
#         self.trend_rotation_prox_controller = TrendRotationProXController(config["trend_rotation_prox"])
#
#     async def create_actions_proposal(self) -> List[CreateExecutorAction]:
#         return await self.trend_rotation_prox_controller.process_tick()
#
#     async def stop_actions_proposal(self) -> List[StopExecutorAction]:
#         # Implement logic to stop executors if needed
#         return []
