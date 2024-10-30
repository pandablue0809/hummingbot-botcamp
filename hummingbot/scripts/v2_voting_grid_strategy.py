import importlib
import inspect
import os
import time
from decimal import Decimal
from typing import Dict, List, Optional, Tuple, Any, Set
import functools

import numpy as np
import pandas as pd
import yaml
from pydantic import Field

from hummingbot.client import settings
from hummingbot.client.config.config_data_types import ClientFieldData
from hummingbot.core.clock import Clock
from hummingbot.core.data_type.common import OrderType, PriceType, TradeType, Decimal
from hummingbot.core.event.events import OrderFilledEvent, BuyOrderCompletedEvent, SellOrderCompletedEvent, OrderCancelledEvent, OrderExpiredEvent, MarketOrderFailureEvent
from hummingbot.data_feed.candles_feed.data_types import CandlesConfig
from hummingbot.exceptions import InvalidController
from hummingbot.strategy.market_trading_pair_tuple import MarketTradingPairTuple
from hummingbot.strategy.strategy_v2_base import StrategyV2Base, StrategyV2ConfigBase\

from controllers.directional_trading.grid_system import grid_system
from controllers.directional_trading.smoothing import get_smoothing_function, available_smoothing_functions
from controllers.directional_trading.vzo_prox import VZOProXConfig, VZOProXController, vzo_prox
from controllers.directional_trading.ten_x_bars_prox import TenXBarsProXConfig, TenXBarsProXController, ten_x_bars_prox
from controllers.directional_trading.madhatter_prox import MadHatterProXConfig, MadHatterProXController, mad_hatter_prox
from controllers.directional_trading.sniperpro import SniperProConfig, SniperProController, sniper_pro
from controllers.directional_trading.reversals_prox import ReversalsProXConfig, ReversalsProXController, reversals_prox
from controllers.directional_trading.hurst_channels_prox import HurstChannelsProXController, HurstChannelsProXConfig
from hummingbot.strategy_v2.controllers import ControllerConfigBase, MarketMakingControllerConfigBase, \
    DirectionalTradingControllerConfigBase


class VotingGridStrategyConfig(StrategyV2ConfigBase):
    script_file_name: str = Field(default_factory=lambda: os.path.basename(__file__))
    connector: str = Field(default="binance", client_data=ClientFieldData(prompt=lambda mi: "Enter the exchange connector"))
    trading_pair: str = Field(default="BTC-USDT", client_data=ClientFieldData(prompt=lambda mi: "Enter the trading pair"))
    grid_type: str = Field(default="auto", client_data=ClientFieldData(prompt=lambda mi: "Select grid type (auto, fib)"))
    grid_smooth_type: str = Field(default="ema", client_data=ClientFieldData(prompt=lambda mi: f"Select smoothing function for grid {list(available_smoothing_functions().keys())}"))
    sniper_pro_smooth_type: str = Field(default="jurik", client_data=ClientFieldData(prompt=lambda mi: f"Select smoothing function for sniper_pro {list(available_smoothing_functions().keys())}"))
    bars_back: int = Field(default=1000, client_data=ClientFieldData(prompt=lambda mi: "Bars Back for Highest/Lowest"))
    grid_params: str = Field(default="BTCUSD", client_data=ClientFieldData(prompt=lambda mi: "Grid Preset"))
    grid_section: int = Field(default=14, client_data=ClientFieldData(prompt=lambda mi: "Grid Section"))
    subsection: str = Field(default="none", client_data=ClientFieldData(prompt=lambda mi: "Subsection"))
    calculation: str = Field(default="arithmetic", client_data=ClientFieldData(prompt=lambda mi: "Calculation Mode"))
    projection_mode: str = Field(default="basic", client_data=ClientFieldData(prompt=lambda mi: "Projection Mode"))
    display_levels: int = Field(default=4, client_data=ClientFieldData(prompt=lambda mi: "Number of Display Levels"))
    start: Decimal = Field(default=Decimal("63.00"), client_data=ClientFieldData(prompt=lambda mi: "Start Value for Grid"))
    stop: Decimal = Field(default=Decimal("1163.00"), client_data=ClientFieldData(prompt=lambda mi: "Stop Value for Grid"))
    projection_point: Decimal = Field(default=Decimal("0.8346"), client_data=ClientFieldData(prompt=lambda mi: "Projection Point"))
    dca_multiplier: Decimal = Field(default=Decimal("1.1"), client_data=ClientFieldData(prompt=lambda mi: "DCA Multiplier"))
    trailing_stop: Decimal = Field(default=Decimal("0.05"), client_data=ClientFieldData(prompt=lambda mi: "Trailing stop percentage"))
    order_amount: Decimal = Field(default=Decimal("0.01"), client_data=ClientFieldData(prompt=lambda mi: "Base order amount"))
    leverage: int = Field(default=1, client_data=ClientFieldData(prompt=lambda mi: "Leverage to use for trading"))
    dca_spreads: List[Decimal] = Field(default=[Decimal("0.01"), Decimal("0.02"), Decimal("0.03"), Decimal("0.04")])
    dca_amounts_pct: List[Decimal] = Field(default=[Decimal("0.4"), Decimal("0.3"), Decimal("0.2"), Decimal("0.1")])
    volatility_adjustment_factor: Decimal = Field(default=Decimal("0.5"))
    # controllers_config: List[str] = Field(
    #     default=[
    #              "conf_directional_trading.vzo_prox_1.yml",
    #              "conf_directional_trading.ten_x_bars_prox_1.yml",
    #              "conf_directional_trading.madhatter_prox_1.yml",
    #              "conf_directional_trading.sniperpro_1.yml",
    #              "conf_directional_trading.reversals_prox_1.yml",
    #              "conf_directional_trading.hurst_channels_prox_1.yml"
    #              ],
    #     client_data=ClientFieldData(
    #         is_updatable=True,
    #         prompt_on_new=True,
    #         prompt=lambda mi: "Enter controller configurations (comma-separated file paths), leave it empty if none: "
    #     ))
    #
    # def load_controller_configs(self):
    #     loaded_configs = []
    #     for config_path in self.controllers_config:
    #         full_path = os.path.join(settings.CONTROLLERS_CONF_DIR_PATH, config_path)
    #         with open(full_path, 'r') as file:
    #             config_data = yaml.safe_load(file)
    #
    #         controller_type = config_data.get('controller_type')
    #         controller_name = config_data.get('controller_name')
    # 
    #         if not controller_type or not controller_name:
    #             raise ValueError(f"Missing controller_type or controller_name in {config_path}")
    #
    #         module_path = f"{settings.CONTROLLERS_MODULE}.{controller_type}.{controller_name}"
    #         module = importlib.import_module(module_path)
    #
    #         config_class = next((member for member_name, member in inspect.getmembers(module)
    #                              if inspect.isclass(member) and member not in [ControllerConfigBase,
    #                                                                            MarketMakingControllerConfigBase,
    #                                                                            DirectionalTradingControllerConfigBase]
    #                              and (issubclass(member, ControllerConfigBase))), None)
    #         if not config_class:
    #             raise InvalidController(f"No configuration class found in the module {controller_name}.")
    #
    #         loaded_configs.append(config_class(**config_data))
    #
    #     return loaded_configs

class VotingGridStrategy(StrategyV2Base):
    @classmethod
    def logger(cls):
        return cls.logger()

    def __init__(self, config: VotingGridStrategyConfig):
        super().__init__()
        self.config = config
        self.config.vzo_prox_config = VZOProXConfig
        self.config.ten_x_bars_prox_config = TenXBarsProXConfig
        self.config.madhatter_prox_config = MadHatterProXConfig
        self.config.sniper_pro_config = SniperProConfig
        self.config.reversals_prox_config = ReversalsProXConfig 
        self.config.hurst_channels_prox_config = HurstChannelsProXConfig
        self.market_info: Optional[MarketTradingPairTuple] = None
        self.current_bias: float = 0
        self.minor_bias_add_percent: float = 5
        self.major_bias_add_percent: float = 10
        self.last_candle_timestamp: float = 0
        self.candles_df: Optional[pd.DataFrame] = None
        self.active_orders: Dict[str, Tuple[str, Decimal]] = {}
        self.sniper_pro_controller = SniperProController(config.sniper_pro_config)
        self.vzo_prox_controller = VZOProXController(config.vzo_prox_config)
        self.ten_x_bars_prox_controller = TenXBarsProXController(config.ten_x_bars_prox_config)
        self.madhatter_prox_controller = MadHatterProXController(config.madhatter_prox_config)
        self.reversals_prox_controller = ReversalsProXController(config.reversals_prox_config)
        self.hurst_channels_controller = HurstChannelsProXController(config.hurst_channels_prox_config)

    def start(self, clock: Clock, timestamp: float):
        self.logger().info("Voting Grid Strategy started.")
        try:
            market = self.connectors[self.config.connector]
            trading_pair = self.config.trading_pair
            self.market_info = MarketTradingPairTuple(market, trading_pair, *trading_pair.split("-"))
        except KeyError:
            self.logger().error(f"Connector {self.config.connector} not found. Please check your configuration.")
            return

        self.apply_initial_setting()
        self.initial_portfolio_value = self.get_portfolio_value()

    def stop(self, clock: Clock):
        self.logger().info("Voting Grid Strategy stopped.")
        self.cancel_all_orders()



    async def tick(self, timestamp: float):
        if not self.market_info:
            self.logger().error("Market info not initialized. Skipping tick.")
            return

        if timestamp - self.last_candle_timestamp >= 60:  # Update candles every minute
            await self.update_candles()
            self.last_candle_timestamp = timestamp

        sniper_pro_actions = await self.sniper_pro_controller.process_tick()
        vzo_prox_actions = await self.vzo_prox_controller.process_tick()
        ten_x_bars_prox_actions = await self.ten_x_bars_prox_controller.process_tick()
        madhatter_prox_actions = await self.madhatter_prox_controller.process_tick()
        reversals_prox_actions = await self.reversals_prox_controller.process_tick()

        # Implement voting logic here to decide on final actions based on the signals from all controllers
        final_actions = self.vote_on_actions(sniper_pro_actions, vzo_prox_actions, ten_x_bars_prox_actions, madhatter_prox_actions, reversals_prox_actions)

        for action in final_actions:
            await self.execute_action(action)

    def vote_on_actions(self, *controller_actions):
        # Implement voting logic here
        # This is a placeholder implementation
        all_actions = [action for actions in controller_actions for action in actions]
        return all_actions  # For now, just return all actions

    async def execute_action(self, action):
        if action['type'] == 'create':
            await self.create_order(action)
        elif action['type'] == 'cancel':
            await self.cancel_order(action)

    async def create_order(self, action):
        try:
            order_id = await self.market_info.market.create_order(
                trading_pair=self.market_info.trading_pair,
                order_type=action['order_type'],
                amount=action['amount'],
                price=action['price']
            )
            self.active_orders[order_id] = (action['order_type'], action['price'])
            self.logger().info(f"Created order {order_id}: {action['order_type']} {action['amount']} @ {action['price']}")
        except Exception as e:
            self.logger().error(f"Error creating order: {e}", exc_info=True)

    async def cancel_order(self, action):
        try:
            await self.market_info.market.cancel_order(action['order_id'])
            if action['order_id'] in self.active_orders:
                del self.active_orders[action['order_id']]
            self.logger().info(f"Cancelled order {action['order_id']}")
        except Exception as e:
            self.logger().error(f"Error cancelling order: {e}", exc_info=True)

    async def update_candles(self):
        try:
            candles = self.market_info.market.get_candles(
                trading_pair=self.config.trading_pair,
                interval="1h",
                max_records=self.config.bars_back
            )
            self.candles_df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            self.candles_df['timestamp'] = pd.to_datetime(self.candles_df['timestamp'], unit='ms')
            self.candles_df.set_index('timestamp', inplace=True)
        except Exception as e:
            self.logger().error(f"Error fetching candles: {e}", exc_info=True)

    def process_signals(self, price: Decimal):
        try:
            df_hash = hash(self.candles_df.to_string())
            grid_buys, grid_sells, grid_lines = self.calculate_grid_signals(df_hash)

            minor_buy, minor_sell, major_buy, major_sell, buy_trail, sell_trail, adaptive_z_cross_down, adaptive_z_cross_up = sniper_pro(
                self.candles_df, 
                smooth_type=self.config.sniper_pro_smooth_type
            )

            vzo_signals = vzo_prox(self.candles_df, self.config.vzo_prox_config)
            ten_x_signals = ten_x_bars_prox(self.candles_df, self.config.ten_x_bars_prox_config)
            mh_signals = mad_hatter_prox(self.candles_df, self.config.madhatter_prox_config)
            reversals_signals = reversals_prox(self.candles_df, self.config.reversals_prox_config)

            self._process_buy_signals(price, grid_buys, minor_buy, major_buy, buy_trail, 
                                      vzo_signals, ten_x_signals, mh_signals, adaptive_z_cross_up, reversals_signals)
            self._process_sell_signals(price, grid_sells, minor_sell, major_sell, sell_trail, 
                                       vzo_signals, ten_x_signals, mh_signals, adaptive_z_cross_down, reversals_signals)

            self._adjust_bias_with_channels(price, ten_x_signals[3], ten_x_signals[4], ten_x_signals[5], ten_x_signals[6])

        except Exception as e:
            self.logger().error(f"Error in processing signals: {e}", exc_info=True)

    def _process_buy_signals(self, price: Decimal, grid_buys: pd.Series, minor_buy: pd.Series, major_buy: pd.Series, 
                             buy_trail: pd.Series, vzo_signals: Tuple, ten_x_signals: Tuple, mh_signals: Tuple, 
                             adaptive_z_cross_up: pd.Series, reversals_signals: Dict[str, pd.Series]):
        buy_signals = [
            grid_buys.iloc[-1],
            minor_buy.iloc[-1],
            major_buy.iloc[-1],
            vzo_signals[0].iloc[-1],  # vzo_minor_buy
            vzo_signals[1].iloc[-1],  # vzo_major_buy
            vzo_signals[4].iloc[-1],  # vzo_adaptive_z_buy
            vzo_signals[6].iloc[-1],  # vzo_zero_line_cross_up
            ten_x_signals[0].iloc[-1],  # ten_x_long
            mh_signals[0].iloc[-1],  # mh_high_zone_alert
            mh_signals[2].iloc[-1],  # mh_minor_high_zone_alert
            adaptive_z_cross_up.iloc[-1],  # sniper_pro adaptive_z_cross_up
            reversals_signals["bull_signal"].iloc[-1],
            reversals_signals["o_bull_signal"].iloc[-1]
        ]
        hurst_channels_signals = self.hurst_channels_controller.calculate_channels()
        
        if any(buy_signals):
            self._adjust_bias(True, minor_buy.iloc[-1], major_buy.iloc[-1], vzo_signals, ten_x_signals, mh_signals, reversals_signals, hurst_channels_signals)
            self._execute_dca_buy(price)

    def _process_sell_signals(self, price: Decimal, grid_sells: pd.Series, minor_sell: pd.Series, major_sell: pd.Series, 
                              sell_trail: pd.Series, vzo_signals: Tuple, ten_x_signals: Tuple, mh_signals: Tuple, 
                              adaptive_z_cross_down: pd.Series, reversals_signals: Dict[str, pd.Series]):
        sell_signals = [
            grid_sells.iloc[-1],
            minor_sell.iloc[-1],
            major_sell.iloc[-1],
            vzo_signals[2].iloc[-1],  # vzo_minor_sell
            vzo_signals[3].iloc[-1],  # vzo_major_sell
            vzo_signals[5].iloc[-1],  # vzo_adaptive_z_sell
            vzo_signals[7].iloc[-1],  # vzo_zero_line_cross_down
            ten_x_signals[1].iloc[-1],  # ten_x_short
            mh_signals[1].iloc[-1],  # mh_low_zone_alert
            mh_signals[3].iloc[-1],  # mh_minor_low_zone_alert
            adaptive_z_cross_down.iloc[-1],  # sniper_pro adaptive_z_cross_down
            reversals_signals["bear_signal"].iloc[-1],
            reversals_signals["o_bear_signal"].iloc[-1]
        ]
        hurst_channels_signals = self.hurst_channels_controller.calculate_channels()
        
        if any(sell_signals):
            self._adjust_bias(False, minor_sell.iloc[-1], major_sell.iloc[-1], vzo_signals, ten_x_signals, mh_signals, reversals_signals, hurst_channels_signals)
            self._execute_dca_sell(price)

    def _adjust_bias(self, is_buy: bool, minor_signal: bool, major_signal: bool, 
                     vzo_signals: Tuple, ten_x_signals: Tuple, mh_signals: Tuple, 
                     reversals_signals: Dict[str, pd.Series], hurst_channels_signals: Dict[str, pd.Series]):
        direction = 1 if is_buy else -1
        if minor_signal:
            self.current_bias += direction * self.minor_bias_add_percent
        if major_signal:
            self.current_bias += direction * self.major_bias_add_percent
        if vzo_signals[0].iloc[-1] or vzo_signals[2].iloc[-1]:  # vzo_minor_buy or vzo_minor_sell
            self.current_bias += direction * (self.minor_bias_add_percent * 0.5)
        if vzo_signals[1].iloc[-1] or vzo_signals[3].iloc[-1]:  # vzo_major_buy or vzo_major_sell
            self.current_bias += direction * (self.major_bias_add_percent * 0.5)
        if ten_x_signals[0].iloc[-1] or ten_x_signals[1].iloc[-1]:  # ten_x_long or ten_x_short
            self.current_bias += direction * (self.major_bias_add_percent * 0.75)
        if ten_x_signals[2].iloc[-1]:  # ten_x_volume_strength
            self.current_bias += direction * (self.minor_bias_add_percent * 0.25)
        if mh_signals[0].iloc[-1] or mh_signals[1].iloc[-1]:  # mh_high_zone_alert or mh_low_zone_alert
            self.current_bias += direction * (self.major_bias_add_percent * 0.8)
        if mh_signals[2].iloc[-1] or mh_signals[3].iloc[-1]:  # mh_minor_high_zone_alert or mh_minor_low_zone_alert
            self.current_bias += direction * (self.minor_bias_add_percent * 0.4)
        if reversals_signals["bull_signal"].iloc[-1] or reversals_signals["obull_signal"].iloc[-1]:
            self.current_bias += direction * (self.major_bias_add_percent * 0.75)
        if reversals_signals["bear_signal"].iloc[-1] or reversals_signals["obear_signal"].iloc[-1]:
            self.current_bias += direction * (self.major_bias_add_percent * 0.75)

        # Hurst Channels adjustments
        if hurst_channels_signals['total_upward_cross'].iloc[-1] > 0:
            self.current_bias += direction * (self.major_bias_add_percent * 0.75)
        if hurst_channels_signals['total_downward_cross'].iloc[-1] > 0:
            self.current_bias -= direction * (self.major_bias_add_percent * 0.75)

        # Additional adjustments based on channel positions
        price = self.get_current_price()
        for i in range(1, 4):
            if price > hurst_channels_signals[f'extend{i}_upper'].iloc[-1]:
                self.current_bias += direction * (self.minor_bias_add_percent * 0.25 * i)
            elif price < hurst_channels_signals[f'extend{i}_lower'].iloc[-1]:
                self.current_bias -= direction * (self.minor_bias_add_percent * 0.25 * i)

        self.current_bias = max(min(self.current_bias, 100), -100)

    def _adjust_bias_with_channels(self, price: Decimal, upper_channel: pd.Series, lower_channel: pd.Series, 
                                   base_line: pd.Series, slope: float):
        current_upper = upper_channel.iloc[-1]
        current_lower = lower_channel.iloc[-1]
        current_base = base_line.iloc[-1]

        if price > current_upper:
            self.current_bias += self.major_bias_add_percent
        elif price < current_lower:
            self.current_bias -= self.major_bias_add_percent
        elif price > current_base:
            self.current_bias += self.minor_bias_add_percent
        elif price < current_base:
            self.current_bias -= self.minor_bias_add_percent

        df_hash = hash(self.candles_df.to_string())
        dynamic_slope_factor = self.calculate_dynamic_slope_factor(df_hash, float(price))

        hurst_channels_signals = self.hurst_channels_controller.calculate_channels()
        hurst_slope = (hurst_channels_signals['extend3'].iloc[-1] - hurst_channels_signals['extend3'].iloc[-10]) / 10

        combined_slope = (slope + hurst_slope) / 2
        self.current_bias += combined_slope * dynamic_slope_factor
        self.current_bias = max(min(self.current_bias, 100), -100)

        self.logger().info(f"Combined Slope impact on bias: {combined_slope * dynamic_slope_factor:.2f} (Combined Slope: {combined_slope:.4f}, Factor: {dynamic_slope_factor:.2f})")

    def _execute_dca_buy(self, price: Decimal):
        total_amount = self.calculate_dynamic_order_amount(self.config.order_amount)
        spreads, amounts = self.calculate_dca_levels(total_amount, is_buy=True)
        
        spreads = [spread * (1 + self.current_bias / 100) for spread in spreads]
        
        for spread, amount in zip(spreads, amounts):
            buy_price = price * (Decimal("1") - spread)
            order_id = self.buy_with_specific_market(
                self.market_info,
                amount,
                OrderType.LIMIT,
                buy_price
            )
            if order_id:
                self.active_orders[order_id] = ("BUY", buy_price)
        
        self.logger().info(f"Placed DCA-style buy orders for {self.config.trading_pair} with bias adjustment: {self.current_bias:.2f}%")

    def _execute_dca_sell(self, price: Decimal):
        total_amount = self.calculate_dynamic_order_amount(self.config.order_amount)
        spreads, amounts = self.calculate_dca_levels(total_amount, is_buy=False)
        
        spreads = [spread * (1 - self.current_bias / 100) for spread in spreads]
        
        for spread, amount in zip(spreads, amounts):
            sell_price = price * (Decimal("1") + spread)
            order_id = self.sell_with_specific_market(
                self.market_info,
                amount,
                OrderType.LIMIT,
                sell_price
            )
            if order_id:
                self.active_orders[order_id] = ("SELL", sell_price)
        
        self.logger().info(f"Placed DCA-style sell orders for {self.config.trading_pair} with bias adjustment: {self.current_bias:.2f}%")

    def calculate_dca_levels(self, total_amount: Decimal, is_buy: bool) -> Tuple[List[Decimal], List[Decimal]]:
        spreads = self.config.dca_spreads
        amounts_pct = self.config.dca_amounts_pct
        
        total_pct = sum(amounts_pct)
        amounts_pct = [a / total_pct for a in amounts_pct]
        
        amounts = [total_amount * pct for pct in amounts_pct]
        
        vzo_adjustment = self._calculate_vzo_adjustment()
        ten_x_adjustment = self._calculate_ten_x_adjustment()
        mh_adjustment = self._calculate_madhatter_adjustment()
        
        combined_adjustment = (vzo_adjustment + ten_x_adjustment + mh_adjustment) / 3
        spreads = [spread * (1 + combined_adjustment) for spread in spreads]
        
        return spreads, amounts

    def calculate_dynamic_order_amount(self, base_amount: Decimal) -> Decimal:
        volatility = self.calculate_market_volatility()
        volatility_factor = Decimal("1") + (volatility * self.config.volatility_adjustment_factor)
        return base_amount * volatility_factor

    def calculate_market_volatility(self) -> Decimal:
        if self.candles_df is not None and not self.candles_df.empty:
            returns = self.candles_df['close'].pct_change().dropna()
            volatility = returns.std()
            return Decimal(str(volatility))
        return Decimal("0")

    def _calculate_vzo_adjustment(self) -> Decimal:
        vzo_signals = vzo_prox(self.candles_df, self.config.vzo_prox_config)
        positive_signals = sum(1 for signal in vzo_signals[:8] if signal.iloc[-1] > 0)
        negative_signals = sum(1 for signal in vzo_signals[:8] if signal.iloc[-1] < 0)
        total_signals = 8
        adjustment = (positive_signals - negative_signals) / total_signals
        return Decimal(str(adjustment))

    def _calculate_ten_x_adjustment(self) -> Decimal:
        ten_x_signals = ten_x_bars_prox(self.candles_df, self.config.ten_x_bars_prox_config)
        long_signal = ten_x_signals[0].iloc[-1]
        short_signal = ten_x_signals[1].iloc[-1]
        volume_strength = ten_x_signals[2].iloc[-1]
        
        adjustment = 0
        if long_signal:
            adjustment += 0.5
        if short_signal:
            adjustment -= 0.5
        if volume_strength:
            adjustment *= 1.5
        
        return Decimal(str(adjustment))

    def _calculate_madhatter_adjustment(self) -> Decimal:
        mh_signals = mad_hatter_prox(self.candles_df, self.config.madhatter_prox_config)
        high_zone_alert = mh_signals[0].iloc[-1]
        low_zone_alert = mh_signals[1].iloc[-1]
        minor_high_zone_alert = mh_signals[2].iloc[-1]
        minor_low_zone_alert = mh_signals[3].iloc[-1]
        
        adjustment = 0
        if high_zone_alert:
            adjustment += 0.5
        if low_zone_alert:
            adjustment -= 0.5
        if minor_high_zone_alert:
            adjustment += 0.25
        if minor_low_zone_alert:
            adjustment -= 0.25
        
        return Decimal(str(adjustment))

    def calculate_dynamic_slope_factor(self, df_hash, price):
        upper_channel = self.candles_df['upper_channel'].iloc[-1]
        lower_channel = self.candles_df['lower_channel'].iloc[-1]
        base_line = self.candles_df['base_line'].iloc[-1]
        
        channel_width = upper_channel - lower_channel
        price_deviation = abs(price - base_line)
        
        relative_deviation = price_deviation / channel_width if channel_width != 0 else 0
        
        base_slope_factor = 5
        max_slope_factor = 15
        
        if relative_deviation > 1:
            return max_slope_factor
        else:
            return base_slope_factor + (max_slope_factor - base_slope_factor) * relative_deviation

    def apply_initial_setting(self):
        if self.is_perpetual():
            self.market_info.market.set_leverage(self.config.leverage)
        self.cancel_all_orders()

    def is_perpetual(self) -> bool:
        return hasattr(self.market_info.market, "set_leverage")

    def cancel_all_orders(self):
        for order_id in self.active_orders.keys():
            self.cancel(self.market_info, order_id)
        self.active_orders.clear()

    def did_complete_buy_order(self, order_completed_event: BuyOrderCompletedEvent):
        order_id = order_completed_event.order_id
        if order_id in self.active_orders:
            del self.active_orders[order_id]
        self.logger().info(f"Buy order completed: {order_completed_event}")

    def did_complete_sell_order(self, order_completed_event: SellOrderCompletedEvent):
        order_id = order_completed_event.order_id
        if order_id in self.active_orders:
            del self.active_orders[order_id]
        self.logger().info(f"Sell order completed: {order_completed_event}")

    def did_cancel_order(self, cancelled_event):
        self.logger().info(f"Order cancelled: {cancelled_event}")

    def did_fail_order(self, order_failed_event):
        self.logger().error(f"Order failed: {order_failed_event}")

    def did_expire_order(self, expired_event):
        self.logger().info(f"Order expired: {expired_event}")

    def did_complete_funding_payment(self, funding_payment_completed_event):
        self.logger().info(f"Funding payment completed: {funding_payment_completed_event}")

    def format_status(self) -> str:
        if not self.market_info:
            return "Strategy not started."
        
        lines = []
        mid_price = self.market_info.get_mid_price()
        base, quote = self.market_info.trading_pair.split("-")
        base_balance = self.market_info.base_balance
        quote_balance = self.market_info.quote_balance

        lines.extend([
            f"Exchange: {self.market_info.market.name}",
            f"Trading Pair: {self.market_info.trading_pair}",
            f"Current mid price: {mid_price:.8f}",
            f"Current bias: {self.current_bias:.2f}%",
            f"Base Asset: {base_balance:.8f} {base}",
            f"Quote Asset: {quote_balance:.8f} {quote}",
            f"Active Orders:",
        ])

        for order_id, (side, price) in self.active_orders.items():
            lines.append(f"  {side} order at {price:.8f}")

        hurst_channels_status = self.hurst_channels_controller.to_format_status()
        lines.extend(["", "Hurst Channels:"] + hurst_channels_status)

        return "\n".join(lines)

    def get_price_type(self, price_type_str: str) -> PriceType:
        if price_type_str == "mid_price":
            return PriceType.MidPrice
        elif price_type_str == "best_bid":
            return PriceType.BestBid
        elif price_type_str == "best_ask":
            return PriceType.BestAsk
        elif price_type_str == "last_price":
            return PriceType.LastTrade
        elif price_type_str == 'last_own_trade_price':
            return PriceType.LastOwnTrade
        else:
            raise ValueError(f"Unrecognized price type string {price_type_str}.")

    def get_order_price(self, trading_pair: str, is_buy: bool, amount: Decimal, price_type=PriceType.MidPrice) -> Decimal:
        if price_type == PriceType.MidPrice:
            price = self.get_mid_price(trading_pair)
        elif price_type == PriceType.BestBid:
            price = self.get_price(trading_pair, is_buy=False)
        elif price_type == PriceType.BestAsk:
            price = self.get_price(trading_pair, is_buy=True)
        elif price_type == PriceType.LastTrade:
            price = self.get_last_price(trading_pair)
        else:
            raise ValueError(f"Unrecognized price type {price_type}")
        return price

    def get_mid_price(self, trading_pair: str) -> Decimal:
        return self.market_info.get_mid_price()

    def get_last_price(self, trading_pair: str) -> Decimal:
        return self.market_info.get_last_price()

    def get_available_balance(self, asset: str) -> Decimal:
        return self.market_info.market.get_available_balance(asset)

    def execute_rebalance(self):
        """
        Execute the rebalancing logic. This method should be called periodically.
        """
        if self.market_info is None:
            return

        current_value = self.calculate_total_value()
        target_value = self.calculate_target_value()

        if current_value < target_value:
            amount_to_buy = (target_value - current_value) / self.get_mid_price(self.market_info.trading_pair)
            self._execute_dca_buy(self.get_mid_price(self.market_info.trading_pair))
        elif current_value > target_value:
            amount_to_sell = (current_value - target_value) / self.get_mid_price(self.market_info.trading_pair)
            self._execute_dca_sell(self.get_mid_price(self.market_info.trading_pair))

    def calculate_total_value(self) -> Decimal:
        """
        Calculate the total value of the current position.
        """
        base_amount = self.get_available_balance(self.market_info.base_asset)
        quote_amount = self.get_available_balance(self.market_info.quote_asset)
        mid_price = self.get_mid_price(self.market_info.trading_pair)
        return base_amount * mid_price + quote_amount

    def calculate_target_value(self) -> Decimal:
        """
        Calculate the target value based on the strategy's signals and current market conditions.
        This is a placeholder and should be implemented based on your strategy's specific logic.
        """
        # Placeholder implementation
        return self.calculate_total_value() * (1 + self.current_bias / 100)

    def update_strategy_parameters(self, new_params: Dict):
        """Update strategy parameters dynamically."""
        for key, value in new_params.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                self.logger().info(f"Updated parameter {key} to {value}")
            else:
                self.logger().warning(f"Parameter {key} not found in strategy configuration")

    def execute_trailing_stop(self):
        """Execute a trailing stop to protect profits or limit losses."""
        current_price = self.get_mid_price(self.market_info.trading_pair)
        position = self.get_available_balance(self.market_info.base_asset)

        if position > 0:  # Long position
            stop_price = current_price * (1 - self.config.trailing_stop)
            if current_price <= stop_price:
                self._execute_dca_sell(current_price)
        elif position < 0:  # Short position
            stop_price = current_price * (1 + self.config.trailing_stop)
            if current_price >= stop_price:
                self._execute_dca_buy(current_price)

    def update_risk_management(self):
        """Update risk management parameters based on market conditions."""
        volatility = self.calculate_market_volatility()
        self.config.order_amount = self.config.order_amount * (1 / (1 + volatility))
        self.config.trailing_stop = max(self.config.trailing_stop, volatility * 2)

    def log_strategy_status(self):
        """Log the current status of the strategy."""
        self.logger().info(self.format_status())

    def on_stop(self):
        """Called when the strategy is stopped."""
        self.cancel_all_orders()
        self.log_strategy_status()

    def on_tick(self):
        """
        Called on each clock tick. This is where the main logic of the strategy should be implemented.
        """
        if not self.market_info:
            return

        self.process_signals(self.get_mid_price(self.market_info.trading_pair))
        self.execute_rebalance()
        self.execute_trailing_stop()
        self.update_risk_management()
        self.log_strategy_status()

    # Event listeners
    def did_fill_order(self, event: OrderFilledEvent):
        """Called when an order is filled (fully or partially)."""
        self.logger().info(f"Order filled. Order ID: {event.order_id}, "
                           f"Trade ID: {event.trade_id}, "
                           f"Trading pair: {event.trading_pair}, "
                           f"Price: {event.price}, "
                           f"Amount: {event.amount}")
        
        # Update strategy state based on the fill
        if event.trade_type == TradeType.BUY:
            self.update_strategy_state_on_buy(event)
        elif event.trade_type == TradeType.SELL:
            self.update_strategy_state_on_sell(event)

    def update_strategy_state_on_buy(self, event: OrderFilledEvent):
        """Update strategy state when a buy order is filled."""
        # Implement logic to update strategy state after a buy
        pass

    def update_strategy_state_on_sell(self, event: OrderFilledEvent):
        """Update strategy state when a sell order is filled."""
        # Implement logic to update strategy state after a sell
        pass

    def did_complete_buy_order(self, event: BuyOrderCompletedEvent):
        """Called when a buy order is completely filled."""
        self.logger().info(f"Buy order completed. Order ID: {event.order_id}, "
                           f"Trading pair: {event.trading_pair}, "
                           f"Price: {event.price}, "
                           f"Amount: {event.base_asset_amount}")
        
        if event.order_id in self.active_orders:
            del self.active_orders[event.order_id]
        
        # Implement any additional logic needed when a buy order is completed

    def did_complete_sell_order(self, event: SellOrderCompletedEvent):
        """Called when a sell order is completely filled."""
        self.logger().info(f"Sell order completed. Order ID: {event.order_id}, "
                           f"Trading pair: {event.trading_pair}, "
                           f"Price: {event.price}, "
                           f"Amount: {event.base_asset_amount}")
        
        if event.order_id in self.active_orders:
            del self.active_orders[event.order_id]
        
        # Implement any additional logic needed when a sell order is completed

    def did_cancel_order(self, event: OrderCancelledEvent):
        """Called when an order is cancelled."""
        self.logger().info(f"Order cancelled. Order ID: {event.order_id}")
        
        if event.order_id in self.active_orders:
            del self.active_orders[event.order_id]

    def did_fail_order(self, event: MarketOrderFailureEvent):
        """Called when an order fails."""
        self.logger().error(f"Order failed. Order ID: {event.order_id}, Error: {event.error_description}")
        
        if event.order_id in self.active_orders:
            del self.active_orders[event.order_id]

    def did_expire_order(self, event: OrderExpiredEvent):
        """Called when an order expires."""
        self.logger().info(f"Order expired. Order ID: {event.order_id}")
        
        if event.order_id in self.active_orders:
            del self.active_orders[event.order_id]

