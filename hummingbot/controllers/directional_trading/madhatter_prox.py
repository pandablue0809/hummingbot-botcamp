import pandas as pd
from typing import Tuple, List, Union
from decimal import Decimal
from pydantic import Field

from hummingbot.client.config.config_data_types import ClientFieldData
from hummingbot.strategy_v2.models.executor_actions import CreateExecutorAction, StopExecutorAction
from hummingbot.data_feed.candles_feed.candles_factory import CandlesFactory
from hummingbot.data_feed.candles_feed.data_types import CandlesConfig
from controllers.directional_trading.smoothing import get_smoothing_function
from hummingbot.strategy_v2.controllers.directional_trading_controller_base import (
    DirectionalTradingControllerBase,
    DirectionalTradingControllerConfigBase,
)

class MadHatterProXConfig(DirectionalTradingControllerConfigBase):
    controller_name: str = "madhatter_prox"
    candles_config: List[CandlesConfig] = []
    atr_factor: float = Field(default=1.618, client_data=ClientFieldData(prompt=lambda msg: "Enter the ATR Factor"))
    higher_hedge_period_1: str = Field(default="1W", client_data=ClientFieldData(prompt=lambda msg: "Enter Higher Hedge Period 1"))
    higher_hedge_period_2: str = Field(default="1M", client_data=ClientFieldData(prompt=lambda msg: "Enter Higher Hedge Period 2"))
    higher_hedge_period_3: str = Field(default="1D", client_data=ClientFieldData(prompt=lambda msg: "Enter Higher Hedge Period 3"))
    length: int = Field(default=14, client_data=ClientFieldData(prompt=lambda msg: "Enter the Length"))
    smooth_type: str = Field(default="Jurik Moving Average", client_data=ClientFieldData(prompt=lambda msg: "Enter the Smoothing Type"))
    phase: float = Field(default=0, client_data=ClientFieldData(prompt=lambda msg: "Enter the Jurik Phase"))
    power: float = Field(default=2, client_data=ClientFieldData(prompt=lambda msg: "Enter the Jurik Power"))
    connector: str = Field(default="binance", client_data=ClientFieldData(prompt=lambda msg: "Enter the exchange connector", prompt_on_new=True))
    trading_pair: str = Field(default="BTC-USDT", client_data=ClientFieldData(prompt=lambda msg: "Enter the trading pair", prompt_on_new=True))

    # ??
    # candle_interval: str
    # bars_back: int
    order_amount: Decimal = Field(default=Decimal("0.1"), client_data=ClientFieldData(prompt=lambda msg: "Enter the order amount"))

def mad_hatter_prox(
    df: pd.DataFrame,
    config: MadHatterProXConfig
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Calculate MadHatter ProX signals.
    :param df: DataFrame with OHLCV data
    :param config: MadHatterProXConfig object
    :return: high_zone_alert, low_zone_alert, minor_high_zone_alert, minor_low_zone_alert
    """
    smoothing_func = get_smoothing_function(config.smooth_type)

    def apply_smoothing(series: pd.Series, length: int) -> pd.Series:
        if config.smooth_type == "Jurik Moving Average":
            return smoothing_func(series, length, phase=config.phase, power=config.power)
        return smoothing_func(series, length)

    def calculate_hedge_zones(data: pd.DataFrame, period: str) -> Tuple[pd.Series, pd.Series, pd.Series]:
        resampled_data = data.resample(period).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last'
        }).ffill()

        src = (resampled_data['open'] + resampled_data['high'] + resampled_data['low'] + resampled_data['close']) / 4
        v = apply_smoothing(src, config.length)
        tr = pd.concat([
            resampled_data['high'] - resampled_data['low'],
            abs(resampled_data['high'] - resampled_data['close'].shift(1)),
            abs(resampled_data['low'] - resampled_data['close'].shift(1))
        ], axis=1).max(axis=1)
        a = apply_smoothing(tr, config.length)

        upper_hedge_zone = v + config.atr_factor * a
        lower_hedge_zone = v - config.atr_factor * a

        return upper_hedge_zone.reindex(data.index).ffill(), lower_hedge_zone.reindex(data.index).ffill(), v.reindex(data.index).ffill()

    upper_hedge_zone_1, lower_hedge_zone_1, v1 = calculate_hedge_zones(df, config.higher_hedge_period_1)
    upper_hedge_zone_2, lower_hedge_zone_2, v2 = calculate_hedge_zones(df, config.higher_hedge_period_2)
    upper_hedge_zone_3, lower_hedge_zone_3, v3 = calculate_hedge_zones(df, config.higher_hedge_period_3)

    high_zone_alert = (df['high'] > upper_hedge_zone_1) & (df['high'] > upper_hedge_zone_2) & (df['high'] > upper_hedge_zone_3)
    low_zone_alert = (df['low'] < lower_hedge_zone_1) & (df['low'] < lower_hedge_zone_2) & (df['low'] < lower_hedge_zone_3)
    minor_high_zone_alert = (df['high'] > upper_hedge_zone_1) & (df['high'] > upper_hedge_zone_2)
    minor_low_zone_alert = (df['low'] < lower_hedge_zone_1) & (df['low'] < lower_hedge_zone_2)

    return high_zone_alert, low_zone_alert, minor_high_zone_alert, minor_low_zone_alert

class MadHatterProXController(DirectionalTradingControllerBase):
    def __init__(self, config: MadHatterProXConfig, *args, **kwargs):
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
        self.consecutive_high = 0
        self.consecutive_low = 0
        super().__init__(config, *args, **kwargs)

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
        signals = self.get_signals()
        return await self.process_signals(signals)

    def get_signals(self):
        return mad_hatter_prox(self.df, self.config)

    async def process_signals(self, signals) -> List[Union[CreateExecutorAction, StopExecutorAction]]:
        high_zone_alert, low_zone_alert, minor_high_zone_alert, minor_low_zone_alert = signals
        actions = []

        if high_zone_alert.iloc[-1] or minor_high_zone_alert.iloc[-1]:
            self.consecutive_high += 1
            self.consecutive_low = 0
            weight = 1 + min(self.consecutive_high * 0.1, 0.5)  # Max 50% increase
            actions.append(CreateExecutorAction(executor_type="sell", amount=self.config.order_amount * Decimal(str(weight))))
        elif low_zone_alert.iloc[-1] or minor_low_zone_alert.iloc[-1]:
            self.consecutive_low += 1
            self.consecutive_high = 0
            weight = 1 + min(self.consecutive_low * 0.1, 0.5)  # Max 50% increase
            actions.append(CreateExecutorAction(executor_type="buy", amount=self.config.order_amount * Decimal(str(weight))))
        else:
            self.consecutive_high = 0
            self.consecutive_low = 0

        return actions

    def to_format_status(self) -> List[str]:
        return [
            f"ATR Factor: {self.config.atr_factor}",
            f"Higher Hedge Period 1: {self.config.higher_hedge_period_1}",
            f"Higher Hedge Period 2: {self.config.higher_hedge_period_2}",
            f"Higher Hedge Period 3: {self.config.higher_hedge_period_3}",
            f"Length: {self.config.length}",
            f"Smoothing type: {self.config.smooth_type}",
        ]

# class MadHatterProXStrategy(GenericV2StrategyWithCashOut):
#     def __init__(self, config: Dict):
#         super().__init__(config)
#         self.mad_hatter_prox_controller = MadHatterProXController(config["mad_hatter_prox"])

#     async def create_actions_proposal(self) -> List[CreateExecutorAction]:
#         return await self.mad_hatter_prox_controller.process_tick()

#     async def stop_actions_proposal(self) -> List[StopExecutorAction]:
#         # Implement logic to stop executors if needed
#         return []
