import pandas as pd
import numpy as np
import pandas_ta as ta
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

def sniper_pro(
    df: pd.DataFrame,
    pro_length: int = 28,
    overbought_oversold: float = 1.386,
    dmi_length: int = 14,
    adx_thresh: int = 20,
    volavg_length: int = 20,
    vol_per_thresh: int = 50,
    smooth_type: str = 'jurik',
    phase: float = 0,
    power: float = 2,
    sho_vol_str_yel: bool = True,
    generate_alerts: bool = False,
    alert_requires_aim: bool = True,
    data_sample: int = 55,
    pcnt_above: int = 88,
    pcnt_below: int = 88
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]: 
    """
    Calculate Sniper Pro signals. This function can be used directly by the Voting Grid Strategy.
    :param df: DataFrame with OHLCV data
    :param pro_length: pro length
    :param overbought_oversold: overbought/oversold threshold
    :param dmi_length: DMI length
    :param adx_thresh: ADX threshold
    :param volavg_length: volume average length
    :param vol_per_thresh: volume percentage threshold
    :param smooth_type: smoothing type
    :param phase: Jurik phase (only used for 'jurik' smoothing)
    :param power: Jurik power (only used for 'jurik' smoothing)
    :param sho_vol_str_yel: Show Volume Strength on Yellow
    :param generate_alerts: Generate alerts
    :param alert_requires_aim: Alert requires AIM
    :param data_sample: Dynamic Zone Sample Length
    :param pcnt_above: DZ Hi is Above X% of Sample
    :param pcnt_below: DZ Lo is Below X% of Sample
    :return: minor buy, minor sell, major buy, major sell, buy trail, sell trail, adaptive_z_cross_down, adaptive_z_cross_up signals
    """
    dmi_length = max(1, dmi_length)
    adx_thresh = max(1, adx_thresh)
    volavg_length = max(1, volavg_length)
    vol_per_thresh = max(1, vol_per_thresh)
    pcnt_above = max(1, min(100, pcnt_above))
    pcnt_below = max(1, min(100, pcnt_below))

    try:
        smoothing_func = get_smoothing_function(smooth_type)
    except ValueError as e:
        raise ValueError(f"Error in selecting smoothing function: {str(e)}")

    def apply_smoothing(series: pd.Series, length: int) -> pd.Series:
        if smooth_type == 'jurik':
            return smoothing_func(series, length, phase=phase, power=power)
        return smoothing_func(series, length)

    stoch_5 = ta.stoch(df['high'], df['low'], df['close'], k=5, d=3)
    stoch_8 = ta.stoch(df['high'], df['low'], df['close'], k=8, d=5)
    stoch_17 = ta.stoch(df['high'], df['low'], df['close'], k=17, d=5)

    k1v = pd.Series([max(-100, min(100, val) - 50) / 50.01 for val in apply_smoothing(stoch_5['STOCHk_5_3_3'], 3)])
    k2v = pd.Series([max(-100, min(100, val) - 50) / 50.01 for val in apply_smoothing(stoch_8['STOCHk_8_5_3'], 5)])
    k3v = pd.Series([max(-100, min(100, val) - 50) / 50.01 for val in apply_smoothing(stoch_17['STOCHk_17_5_3'], 5)])

    def tema2(x: pd.Series, y: int) -> pd.Series:
        ema1 = apply_smoothing(x, y)
        ema2 = apply_smoothing(ema1, y)
        ema3 = apply_smoothing(ema2, y)
        return (3 * ema1) - (3 * ema2) + ema3

    tavg = tema2(df['close'], pro_length)
    savg_length = pro_length // 2
    savg = tavg.shift(savg_length)
    matr = 0.5 * ta.true_range(df['high'], df['low'], df['close'])

    savgstate = pd.Series([1 if row['low'] > (savg[i] + matr[i]) else -1 if row['high'] < (savg[i] - matr[i]) else 0 
                           for i, row in df.iterrows()])

    # ADX-DMI calculations
    upMove = df['high'] - df['high'].shift(1)
    downMove = -(df['low'] - df['low'].shift(1))
    plusDM = pd.Series([val if val > downMove[i] and val > 0 else 0 for i, val in enumerate(upMove)])
    minusDM = pd.Series([val if val > upMove[i] and val > 0 else 0 for i, val in enumerate(downMove)])

    trueRange = apply_smoothing(ta.true_range(df['high'], df['low'], df['close']).fillna(0), dmi_length)
    DIp = (100 * apply_smoothing(plusDM, dmi_length)).fillna(method='ffill') / trueRange
    DIm = (100 * apply_smoothing(minusDM, dmi_length)).fillna(method='ffill') / trueRange
    sum0 = DIp + DIm
    adx = 100 * apply_smoothing(np.abs(DIp - DIm).fillna(0) / sum0.fillna(0).replace(0, 1), dmi_length)
    long = (DIp > DIm) & (adx > adx_thresh)
    short = (DIm > DIp) & (adx > adx_thresh)
    aVol = apply_smoothing(df['volume'], volavg_length)
    pVol = 100 * (df['volume'] - aVol.shift(1)) / df['volume']
    
    hh_sig = (adx.shift(-savg_length) <= adx.shift(-savg_length-1)) | (adx.shift(-savg_length-1) > adx.shift(-savg_length-2))
    hh_sig = hh_sig.replace(True, np.nan).replace(False, 0)
    hh_down = (savgstate * hh_sig.replace(np.nan, 1) == -1).replace(True, 0).replace(False, np.nan)

    countChg = pd.Series([0] * len(df))
    sc = pd.Series([0] * len(df))
    for i in range(1, len(df)):
        if k2v.iloc[i] > 0:
            if k1v.iloc[i] <= k2v.iloc[i] and k1v.iloc[i-1] > k2v.iloc[i-1] and k2v.iloc[i-1] > 0:
                countChg.iloc[i] = -1
            sc.iloc[i] = min(sc.iloc[i-1], 0) + countChg.iloc[i]
        else:
            if k1v.iloc[i] >= k2v.iloc[i] and k1v.iloc[i-1] < k2v.iloc[i-1] and k2v.iloc[i-1] <= 0:
                countChg.iloc[i] = 1
            sc.iloc[i] = max(sc.iloc[i-1], 0) + countChg.iloc[i]

    # F Values
    f3 = pd.Series([0] * len(df))
    for i in range(1, len(df)):
        log_val = 0.5 * (np.log((1 + k3v.iloc[i]) / (1 - k3v.iloc[i])) + f3.iloc[i-1])
        f3.iloc[i] = log_val if not np.isnan(log_val) and log_val is not None else f3.iloc[i-1]

    minor_buy = pd.Series([False] * len(df))
    major_buy = pd.Series([False] * len(df))
    minor_sell = pd.Series([False] * len(df))
    major_sell = pd.Series([False] * len(df))
    buy_trail = pd.Series([np.nan] * len(df))
    sell_trail = pd.Series([np.nan] * len(df))

    for i in range(2, len(df)):
        if np.sign(f3.iloc[i] - f3.iloc[i-1]) > np.sign(f3.iloc[i-1] - f3.iloc[i-2]):
            if f3.iloc[i] < -overbought_oversold:
                minor_buy.iloc[i] = True
            elif f3.iloc[i] < -overbought_oversold and sc.iloc[i] > 1:
                major_buy.iloc[i] = True
        elif np.sign(f3.iloc[i] - f3.iloc[i-1]) < np.sign(f3.iloc[i-1] - f3.iloc[i-2]):
            if f3.iloc[i] > overbought_oversold:
                minor_sell.iloc[i] = True
            elif f3.iloc[i] > overbought_oversold and sc.iloc[i] < -1:
                major_sell.iloc[i] = True

        if np.sign(f3.iloc[i] - f3.iloc[i-1]) > np.sign(f3.iloc[i-1] - f3.iloc[i-2]) and f3.iloc[i] > 0.9:
            buy_trail.iloc[i] = f3.iloc[i-1]
        elif np.sign(f3.iloc[i] - f3.iloc[i-1]) < np.sign(f3.iloc[i-1] - f3.iloc[i-2]) and f3.iloc[i] < -0.9:
            sell_trail.iloc[i] = f3.iloc[i-1]

    # Adaptive Zone calculations
    osc_src = f3
    smpl_above = df['close'].rolling(window=data_sample).apply(lambda x: np.percentile(x, pcnt_above))
    smpl_below = df['close'].rolling(window=data_sample).apply(lambda x: np.percentile(x, 100 - pcnt_below))

    adaptive_z_cross_down = (osc_src <= smpl_above) & (osc_src.shift(1) > smpl_above) & (osc_src.shift(2) - osc_src > 0.01)
    adaptive_z_cross_up = (osc_src >= smpl_below) & (osc_src.shift(1) < smpl_below) & (osc_src.shift(2) - osc_src < -0.01)

    return minor_buy, minor_sell, major_buy, major_sell, buy_trail, sell_trail, adaptive_z_cross_down, adaptive_z_cross_up

class SniperProConfig(DirectionalTradingControllerBase):
    controller_name: str = "sniperpro"
    candles_config: List[CandlesConfig] = []
    connector: str = Field(default="binance", client_data=ClientFieldData(prompt=lambda msg: "Enter the exchange connector: ", prompt_on_new=True))
    trading_pair: str = Field(default="BTC-USDT", client_data=ClientFieldData(prompt=lambda msg: "Enter the trading pair: ", prompt_on_new=True))
    pro_length: int = Field(default=28, client_data=ClientFieldData(prompt=lambda msg: "Enter the pro length: ", prompt_on_new=True))
    overbought_oversold: float = Field(default=1.386, client_data=ClientFieldData(prompt=lambda msg: "Enter the overbought/oversold threshold: ", prompt_on_new=True))
    dmi_length: int = Field(default=14, client_data=ClientFieldData(prompt=lambda msg: "Enter the DMI length: ", prompt_on_new=True))
    adx_thresh: int = Field(default=20, client_data=ClientFieldData(prompt=lambda msg: "Enter the ADX threshold: ", prompt_on_new=True))
    volavg_length: int = Field(default=20, client_data=ClientFieldData(prompt=lambda msg: "Enter the volume average length: ", prompt_on_new=True))
    vol_per_thresh: int = Field(default=50, client_data=ClientFieldData(prompt=lambda msg: "Enter the volume percentage threshold: ", prompt_on_new=True))
    smooth_type: str = Field(default="jurik", client_data=ClientFieldData(prompt=lambda msg: "Enter the smoothing type: ", prompt_on_new=True))
    phase: float = Field(default=0, client_data=ClientFieldData(prompt=lambda msg: "Enter the Jurik phase: ", prompt_on_new=True))
    power: float = Field(default=2, client_data=ClientFieldData(prompt=lambda msg: "Enter the Jurik power: ", prompt_on_new=True))
    data_sample: int = Field(default=55, client_data=ClientFieldData(prompt=lambda msg: "Enter the Dynamic Zone Sample Length: ", prompt_on_new=True))
    pcnt_above: int = Field(default=88, client_data=ClientFieldData(prompt=lambda msg: "Enter the DZ Hi Above X% of Sample: ", prompt_on_new=True))
    pcnt_below: int = Field(default=88, client_data=ClientFieldData(prompt=lambda msg: "Enter the DZ Lo Below X% of Sample: ", prompt_on_new=True))
    candle_interval: str = Field(default="5m", client_data=ClientFieldData(prompt=lambda msg: "Enter the Interval: ", prompt_on_new=True))
    bars_back: int
    order_amount: Decimal = Field(default=Decimal("0.1"), client_data=ClientFieldData(prompt=lambda msg: "Enter the order amount: ", prompt_on_new=True))

class SniperProController(DirectionalTradingControllerConfigBase):
    def __init__(self, config: SniperProConfig, *args, **kwargs):
        self.config = config
        self.df = pd.DataFrame()
        self.max_records = config.pro_length
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

    async def process_tick(self) -> List[Union[CreateExecutorAction, StopExecutorAction]]:
        await self.update_market_data()
        signals = self.get_signals()
        return await self.process_signals(signals)

    def get_signals(self):
        return sniper_pro(self.df, **self.config.dict())

    async def process_signals(self, signals) -> List[Union[CreateExecutorAction, StopExecutorAction]]:
        minor_buy, minor_sell, major_buy, major_sell, buy_trail, sell_trail, adaptive_z_cross_down, adaptive_z_cross_up = signals
        actions = []

        if major_buy.iloc[-1] or (minor_buy.iloc[-1] and adaptive_z_cross_up.iloc[-1]):
            self.consecutive_buy += 1
            self.consecutive_sell = 0
            weight = 1 + min(self.consecutive_buy * 0.1, 0.5)  # Max 50% increase
            actions.append(CreateExecutorAction(executor_type="buy", amount=self.config.order_amount * Decimal(str(weight))))
        elif major_sell.iloc[-1] or (minor_sell.iloc[-1] and adaptive_z_cross_down.iloc[-1]):
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
            f"Pro Length: {self.config.pro_length}",
            f"Overbought/Oversold: {self.config.overbought_oversold}",
            f"DMI Length: {self.config.dmi_length}",
            f"ADX Threshold: {self.config.adx_thresh}",
            f"Smoothing type: {self.config.smooth_type}",
            f"Consecutive Buy Signals: {self.consecutive_buy}",
            f"Consecutive Sell Signals: {self.consecutive_sell}",
        ]

# class SniperProStrategy(GenericV2StrategyWithCashOut):
#     def __init__(self, config: Dict):
#         super().__init__(config)
#         self.sniper_pro_controller = SniperProController(config["sniper_pro"])

#     async def create_actions_proposal(self) -> List[CreateExecutorAction]:
#         return await self.sniper_pro_controller.process_tick()

#     async def stop_actions_proposal(self) -> List[StopExecutorAction]:
#         # Implement logic to stop executors if needed
#         return []