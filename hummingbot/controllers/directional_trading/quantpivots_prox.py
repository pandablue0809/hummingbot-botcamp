import pandas as pd
import numpy as np
from typing import Dict, List, Union
from decimal import Decimal
from pydantic import Field

from hummingbot.client.config.config_data_types import ClientFieldData
from hummingbot.strategy_v2.models.executor_actions import CreateExecutorAction, StopExecutorAction
from hummingbot.data_feed.candles_feed.candles_factory import CandlesFactory
from hummingbot.data_feed.candles_feed.data_types import CandlesConfig
from .smoothing import available_smoothing_functions, get_smoothing_function, all_smoothing_func_names, available_smoothing_functions
from hummingbot.strategy_v2.controllers.directional_trading_controller_base import (
    DirectionalTradingControllerBase,
    DirectionalTradingControllerConfigBase,
)

smoothing_func_msg = f"Enter smoothing function ({', '.join(available_smoothing_functions())})"

class QuantPivotsProXConfig(DirectionalTradingControllerConfigBase):
    connector: str = Field(default="binance", client_data=ClientFieldData(prompt=lambda msg: "Enter the exchange connector: ", prompt_on_new=True))
    trading_pair: str = Field(default="BTC-USDT", client_data=ClientFieldData(prompt=lambda msg: "Enter the trading pair: ", prompt_on_new=True))
    candle_interval: str = Field(default="BTC-USDT", client_data=ClientFieldData(prompt=lambda msg: "Enter the Candle Interval: ", prompt_on_new=True))
    bars_back: int = Field(default=5, client_data=ClientFieldData(prompt=lambda msg: "Enter the bars back: ", prompt_on_new=True))
    analysis_period: str = Field(default="1w", client_data=ClientFieldData(prompt=lambda msg: "Enter the analysis period (e.g., 1d, 1w, 1M)", prompt_on_new=True))
    averaging_period: int = Field(default=30, client_data=ClientFieldData(prompt=lambda msg: "Enter the averaging period", prompt_on_new=True))
    year_quarter_averaging_period: int = Field(default=10, client_data=ClientFieldData(prompt=lambda msg: "Enter the year/quarter averaging period", prompt_on_new=True))
    h2l2_standard_deviations: float = Field(default=1.0, client_data=ClientFieldData(prompt=lambda msg: "Enter the H2/L2 standard deviations", prompt_on_new=True))
    order_amount: Decimal = Field(default=Decimal("0.1"), client_data=ClientFieldData(prompt=lambda msg: "Enter the order amount", prompt_on_new=True))
    avail_smoothing_funcs = available_smoothing_functions()
    smoothing_function: str = Field(default="sma", client_data=ClientFieldData(prompt=lambda msg: smoothing_func_msg, prompt_on_new=True))

def quant_pivots_prox(df: pd.DataFrame, config: QuantPivotsProXConfig) -> Dict[str, pd.Series]:
    """
    Calculate Quant Pivots ProX signals.
    :param df: DataFrame with OHLCV data
    :param config: QuantPivotsProXConfig object
    :return: Dictionary of signals
    """
    calculation_period = config.year_quarter_averaging_period if config.analysis_period in ['1M', '3M'] else config.averaging_period
    smoothing_func = get_smoothing_function(config.smoothing_function)

    df['period_open'] = df['open'].resample(config.analysis_period).first()
    df['period_high'] = df['high'].resample(config.analysis_period).max()
    df['period_low'] = df['low'].resample(config.analysis_period).min()
    df['period_close'] = df['close'].resample(config.analysis_period).last()

    df['up'] = 100 * (df['period_high'].shift(1) - df['period_open'].shift(1)) / df['period_close'].shift(1)
    df['down'] = 100 * abs(df['period_open'].shift(1) - df['period_low'].shift(1)) / df['period_close'].shift(1)

    df['ave_up'] = smoothing_func(df['up'], calculation_period)
    df['ave_down'] = smoothing_func(df['down'], calculation_period)

    df['up_sd'] = df['up'].rolling(calculation_period).std() * config.h2l2_standard_deviations
    df['down_sd'] = df['down'].rolling(calculation_period).std() * config.h2l2_standard_deviations

    df['h1'] = df['period_open'] + (df['ave_up'] / 100) * df['period_open']
    df['h2'] = df['period_open'] + ((df['ave_up'] + df['up_sd']) / 100) * df['period_open']
    df['l1'] = df['period_open'] - (df['ave_down'] / 100) * df['period_open']
    df['l2'] = df['period_open'] - ((df['ave_down'] + df['down_sd']) / 100) * df['period_open']

    return {
        "period_open": df['period_open'],
        "h1": df['h1'],
        "h2": df['h2'],
        "l1": df['l1'],
        "l2": df['l2'],
    }

class QuantPivotsProXController(DirectionalTradingControllerBase):
    def __init__(self, config: QuantPivotsProXConfig):
        self.config = config
        self.df = pd.DataFrame()
        self.candles_config = [CandlesConfig(
            connector=config.connector,
            trading_pair=config.trading_pair,
            interval=config.candle_interval,
            max_records=config.bars_back
        )]
        self.candles_factory = CandlesFactory(self.candles_config)
        self.scan_h2 = 0
        self.scan_h1 = 0
        self.scan_l2 = 0
        self.scan_l1 = 0
        super().__init__(config)

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
        signals = quant_pivots_prox(self.df, self.config)
        return await self.process_signals(signals)

    async def process_signals(self, signals: Dict[str, pd.Series]) -> List[Union[CreateExecutorAction, StopExecutorAction]]:
        actions = []
        current_price = self.df['close'].iloc[-1]

        # Define DCA levels and corresponding order sizes
        dca_levels = {
            'h2': Decimal('1.0'),
            'h1': Decimal('0.75'),
            'l1': Decimal('0.75'),
            'l2': Decimal('1.0')
        }

        base_order_amount = self.config.order_amount

        for level, multiplier in dca_levels.items():
            level_price = signals[level].iloc[-1]
            order_amount = base_order_amount * multiplier

            if level in ['h2', 'h1'] and current_price <= level_price:
                # Sell orders at resistance levels
                actions.append(CreateExecutorAction(
                    executor_type="sell",
                    amount=order_amount,
                    order_type="limit",
                    price=Decimal(str(level_price))
                ))
            elif level in ['l1', 'l2'] and current_price >= level_price:
                # Buy orders at support levels
                actions.append(CreateExecutorAction(
                    executor_type="buy",
                    amount=order_amount,
                    order_type="limit",
                    price=Decimal(str(level_price))
                ))

        # Update scan values
        self.scan_h2 = 1 if current_price > signals['h2'].iloc[-1] else 0
        self.scan_h1 = 1 if current_price > signals['h1'].iloc[-1] else 0
        self.scan_l2 = 1 if current_price < signals['l2'].iloc[-1] else 0
        self.scan_l1 = 1 if current_price < signals['l1'].iloc[-1] else 0

        return actions

    def to_format_status(self) -> List[str]:
        return [
            f"Analysis Period: {self.config.analysis_period}",
            f"Averaging Period: {self.config.averaging_period}",
            f"H2/L2 Standard Deviations: {self.config.h2l2_standard_deviations}",
            f"Order Amount: {self.config.order_amount}",
            f"Smoothing Function: {self.config.smoothing_function}",
        ]
#
# class QuantPivotsProXStrategy(GenericV2StrategyWithCashOut):
#     def __init__(self, config: Dict):
#         super().__init__(config)
#         self.quant_pivots_prox_controller = QuantPivotsProXController(config["quant_pivots_prox"])
#
#     async def create_actions_proposal(self) -> List[CreateExecutorAction]:
#         return await self.quant_pivots_prox_controller.process_tick()
#
#     async def stop_actions_proposal(self) -> List[StopExecutorAction]:
#         # Implement logic to stop executors if needed
#         return []
