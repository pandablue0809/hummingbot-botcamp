from decimal import Decimal
from typing import List, Tuple, Union

import pandas as pd
import numpy as np
import pandas_ta as ta  # type: ignore
from pydantic import Field

from hummingbot.client.config.config_data_types import ClientFieldData
from hummingbot.core.data_type.common import TradeType
from hummingbot.data_feed.candles_feed.data_types import CandlesConfig
from hummingbot.strategy_v2.executors.position_executor.data_types import PositionExecutorConfig, TripleBarrierConfig
from hummingbot.strategy_v2.models.executor_actions import CreateExecutorAction, StopExecutorAction

from controllers.directional_trading.smoothing import available_smoothing_functions, get_smoothing_function
from hummingbot.strategy_v2.controllers.directional_trading_controller_base import (
    DirectionalTradingControllerBase,
    DirectionalTradingControllerConfigBase,
)

# Utility functions

def calculate_p_vars(grid_section: int, calculation: str, bi: float, ei: float) -> List[float]:
    if grid_section == 1:
        p_vars = [bi, ei]
    else:
        if grid_section == 2:
            p_vars = [ei]
        else:
            if calculation == 'arithmetic':
                p_vars = [bi + 1.6180339887 ** (grid_section - 2) * (ei - bi)]
            else:
                p_vars = [bi * 1.6180339887 ** (grid_section - 2) * (ei / bi)]

        if calculation == 'arithmetic':
            p_vars.append(bi + 1.6180339887 ** (grid_section - 1) * (ei - bi))
        else:
            p_vars.append(bi * 1.6180339887 ** (grid_section - 1) * (ei / bi))

    return p_vars


def calculate_s_vars(display_levels: int, p_vars: List[float]) -> List[float]:
    if display_levels >= 2:
        s_vars = [
            p_vars[0] + .382 * (p_vars[1] - p_vars[0]),
            p_vars[0] + .618 * (p_vars[1] - p_vars[0])
        ]
    else:
        s_vars = [0] * 2
    return s_vars


def calculate_t_vars(display_levels: int, p_vars: List[float], s_vars: List[float]) -> List[float]:
    if display_levels >= 3:
        t_vars = [
            p_vars[0] + .382 * (s_vars[0] - p_vars[0]),
            p_vars[0] + .618 * (s_vars[0] - p_vars[0]),
            s_vars[0] + .382 * (s_vars[1] - s_vars[0]),
            s_vars[0] + .618 * (s_vars[1] - s_vars[0]),
            s_vars[1] + .382 * (p_vars[1] - s_vars[1]),
            s_vars[1] + .618 * (p_vars[1] - s_vars[1])
        ]
    else:
        t_vars = [0] * 6
    return t_vars


def calculate_q_vars(display_levels: int) -> List[float]:
    if display_levels >= 4:
        add_nums = ['p_vars[0]', 't_vars[0]', 't_vars[1]', 's_vars[0]', 't_vars[2]', 't_vars[3]', 's_vars[1]',
                    't_vars[4]', 't_vars[5]']
        sub_nums = [['t_vars[0]', 'p_vars[0]'], ['t_vars[1]', 't_vars[0]'], ['s_vars[0]', 't_vars[1]'],
                    ['t_vars[2]', 's_vars[0]'], ['t_vars[3]', 't_vars[2]'], ['s_vars[1]', 't_vars[3]'],
                    ['t_vars[4]', 's_vars[1]'], ['t_vars[5]', 't_vars[4]'], ['p_vars[1]', 't_vars[5]']]
        c = 0

        q_vars = []
        for i in range(1, 19):
            q_vars.append(
                eval(add_nums[c]) + (.382 if i % 2 != 0 else .618) * (eval(sub_nums[c][0]) - eval(sub_nums[c][1])))
            if i % 2 == 0:
                c += 1
    else:
        q_vars = [0] * 18
    return q_vars


def calculate_n_vars(display_levels: int) -> List[float]:
    if display_levels >= 5:
        add_nums = ['p_vars[0]', 'q_vars[0]', 'q_vars[1]', 't_vars[0]', 'q_vars[2]', 'q_vars[3]', 't_vars[1]',
                    'q_vars[4]', 'q_vars[5]', 's_vars[0]', 'q_vars[6]', 'q_vars[7]', 't_vars[2]', 'q_vars[8]',
                    'q_vars[9]', 't_vars[3]', 'q_vars[10]', 'q_vars[11]', 's_vars[1]', 'q_vars[12]', 'q_vars[13]',
                    't_vars[4]', 'q_vars[14]', 'q_vars[15]', 't_vars[5]', 'q_vars[16]', 'q_vars[17]']
        sub_nums = [['q_vars[0]', 'p_vars[0]'], ['q_vars[1]', 'q_vars[0]'], ['t_vars[0]', 'q_vars[1]'],
                    ['q_vars[2]', 't_vars[0]'], ['q_vars[3]', 'q_vars[2]'], ['t_vars[1]', 'q_vars[3]'],
                    ['q_vars[4]', 't_vars[1]'], ['q_vars[5]', 'q_vars[4]'], ['s_vars[0]', 'q_vars[5]']
            , ['q_vars[6]', 's_vars[0]'], ['q_vars[7]', 'q_vars[6]'], ['t_vars[2]', 'q_vars[7]'],
                    ['q_vars[8]', 't_vars[2]'], ['q_vars[9]', 'q_vars[8]'], ['t_vars[3]', 'q_vars[9]'],
                    ['q_vars[10]', 't_vars[3]'], ['q_vars[11]', 'q_vars[10]'], ['s_vars[1]', 'q_vars[11]']
            , ['q_vars[12]', 's_vars[1]'], ['q_vars[13]', 'q_vars[12]'], ['t_vars[4]', 'q_vars[13]'],
                    ['q_vars[14]', 't_vars[4]'], ['q_vars[15]', 'q_vars[14]'], ['t_vars[5]', 'q_vars[15]'],
                    ['q_vars[16]', 't_vars[5]'], ['q_vars[17]', 'q_vars[16]'], ['p_vars[1]', 'q_vars[17]']]
        c = 0

        n_vars = []
        for i in range(1, 55):
            n_vars.append(
                eval(add_nums[c]) + (.382 if i % 2 != 0 else .618) * (eval(sub_nums[c][0]) - eval(sub_nums[c][1])))
            if i % 2 == 0:
                c += 1
    else:
        n_vars = [0] * 54
    return n_vars


def use_subsection(subsection: str, p_vars: List[float]) -> List[float]:
    if subsection == '1':
        p_vars[1] = .382 * (p_vars[1] - p_vars[0]) + p_vars[0]
    elif subsection == '2':
        p_vars[0] = .382 * (p_vars[1] - p_vars[0]) + p_vars[0]
        p_vars[1] = .618 * (p_vars[1] - p_vars[0]) + p_vars[0]
    elif subsection == '3':
        p_vars[0] = .618 * (p_vars[1] - p_vars[0]) + p_vars[0]
    elif subsection == '1.1':
        p_vars[1] = .101124 * (p_vars[1] - p_vars[0]) + p_vars[0]
    elif subsection == '1.2':
        p_vars[0] = .101124 * (p_vars[1] - p_vars[0]) + p_vars[0]
        p_vars[1] = .196524 * (p_vars[1] - p_vars[0]) + p_vars[0]
    elif subsection == '1.3':
        p_vars[0] = 0.196524 * (p_vars[1] - p_vars[0]) + p_vars[0]
        p_vars[1] = 0.318 * (p_vars[1] - p_vars[0]) + p_vars[0]
    elif subsection == '2.1':
        p_vars[0] = 0.318 * (p_vars[1] - p_vars[0]) + p_vars[0]
        p_vars[1] = 0.4134 * (p_vars[1] - p_vars[0]) + p_vars[0]
    elif subsection == '2.2':
        p_vars[0] = 0.4134 * (p_vars[1] - p_vars[0]) + p_vars[0]
        p_vars[1] = 0.5034 * (p_vars[1] - p_vars[0]) + p_vars[0]
    elif subsection == '2.3':
        p_vars[0] = 0.5034 * (p_vars[1] - p_vars[0]) + p_vars[0]
        p_vars[1] = 0.618 * (p_vars[1] - p_vars[0]) + p_vars[0]
    elif subsection == '3.1':
        p_vars[0] = 0.618 * (p_vars[1] - p_vars[0]) + p_vars[0]
        p_vars[1] = 0.739476 * (p_vars[1] - p_vars[0]) + p_vars[0]
    elif subsection == '3.2':
        p_vars[0] = 0.739476 * (p_vars[1] - p_vars[0]) + p_vars[0]
        p_vars[1] = 0.854076 * (p_vars[1] - p_vars[0]) + p_vars[0]
    elif subsection == '3.3':
        p_vars[0] = 0.854076 * (p_vars[1] - p_vars[0]) + p_vars[0]
    return p_vars


def preset_params(grid_params: str, start: float, stop: float, highest_all: float, lowest_all: float) -> Tuple[
    float, float]:
    preset_dict = {
        'Chart Range': (lowest_all, highest_all),
        'BTCUSD': (0.01, 31.91),
        'BTCUSD 1': (63, 1163),
        'BTCUSD 2': (162, 19804.25),
        'THETAUSDT': (0.03554, 0.589),
        'THETAUSDT 2': (1.46, 3.97),
        'TSLA': (25.52, 194.5),
        'TSLA 2020': (5.10, 38.90),
        '$DJI 2002': (7197.49, 9043.47),
        '$DJI 1932': (40.60, 2746.70),
        'YM 2002': (7174, 9044),
        'DIA': (71.81, 90.44),
        'SPX 2002': (768.63, 954.28),
        'SPX 1932': (4.40, 337.88),
        'ES 2002': (767.25, 955.25),
        'SPY': (77.07, 96.05),
        'COMP 2002': (1108.40, 1521.40),
        'NDX 2001': (795.25, 1155.68),
        'NQ 2002': (797.50, 1157.50),
        'QQQQ': (19.76, 28.79),
        'RUT 2002': (324.90, 413.65),
        'TF 2002': (324.10, 416.30),
        'IWM': (32.3, 41.38),
        'OEX': (387.8, 487.94),
        '$DJT 2002': (1918.12, 3090.07),
        '$DXY': (70.70, 89.62),
        'DX': (71.05, 89.62),
        'EURUSD2000': (0.8227, 0.9596),
        'EURJPY2000': (88.94, 113.75),
        'AUDUSD': (0.4475, 0.8002),
        'ZB': (45.448, 62.032),
        'ZN': (63.947, 87.28125),
        'CL': (9.75, 41.15),
        'NG 1': (1.045, 2.490),
        'NG 2': (1.045, 15.65),
        'GC 1': (252.5, 332.5),
        'GC 2': (252.5, 1033.9),
        'SI': (3.505, 7.5),
        'GBPUSD': (1.3501, 1.7402),
        'NZDUSD': (0.3897, 0.7097)
    }
    return preset_dict.get(grid_params, (start, stop))


# Main utility functions to be used by Voting Grid Strategy

def fibgrid_lines(
        df: pd.DataFrame,
        bars_back: int = 100,
        grid_params: str = 'BTCUSD',
        grid_section: int = 5,
        subsection: str = 'none',
        calculation: str = 'arithmetic',
        projection_mode: str = 'basic',
        display_levels: int = 2,
        start: float = 63,
        stop: float = 1163,
        projection_point: float = 0.8346
) -> Tuple[List[float], List[float], List[float], List[float], List[float]]:
    """
    Calculate Fibonacci grid lines.

    :param df: DataFrame with OHLCV data
    :param bars_back: number of bars to look back
    :param grid_params: grid parameters preset
    :param grid_section: grid section
    :param subsection: subsection options
    :param calculation: calculation mode
    :param projection_mode: projection mode
    :param display_levels: number of display levels
    :param start: start value for manual grid
    :param stop: stop value for manual grid
    :param projection_point: projection point for 'fixed point'
    :return: fire_lines, tree_lines, snow_lines, sky_lines, dm_lines
    """

    highest_all = df['high'].tail(bars_back).max()
    lowest_all = df['low'].tail(bars_back).min()

    begin, end = preset_params(grid_params, start, stop, highest_all, lowest_all)

    if projection_mode == 'high':
        bi, ei = end, end + end - begin
    elif projection_mode == 'fixed point':
        bi, ei = projection_point, projection_point + (end - begin)
    elif projection_mode == '61.8 level':
        bi = begin + .618 * (end - begin)
        ei = bi + end - begin
    else:
        bi, ei = begin, end

    fire_lines = calculate_p_vars(grid_section, calculation, bi, ei)
    fire_lines = use_subsection(subsection, fire_lines)

    tree_lines = calculate_s_vars(display_levels, fire_lines)
    snow_lines = calculate_t_vars(display_levels, fire_lines, tree_lines)
    sky_lines = calculate_q_vars(display_levels, fire_lines, tree_lines, snow_lines)
    dm_lines = calculate_n_vars(display_levels, fire_lines, tree_lines, snow_lines, sky_lines)

    return fire_lines, tree_lines, snow_lines, sky_lines, dm_lines


def auto_grid(
        candles: pd.DataFrame,
        i_len: int = 7,
        i_dir: str = 'neutral',
        i_cool: int = 2,
        i_reset: bool = True,
        i_extr: bool = True,
        mintick: float = 1,
        i_elstx: int = 50,
        i_lz: int = 4,
        i_gi: int = 2,
        i_grids: int = 6,
        smooth_type: str = 'ema',
        **kwargs
) -> Tuple[pd.Series, pd.Series, List[List[float]]]:
    """
    Calculate auto grid lines and signals.

    :param candles: DataFrame with OHLCV data
    :param i_len: smoothing length
    :param i_dir: direction. options=['neutral','up','down']
    :param i_cool: cooldown
    :param i_reset: reset buy/sell index when grids change
    :param i_extr: true if use high/low for signals
    :param mintick: the minimum tick amount for the symbol
    :param i_elstx: elasticity
    :param i_lz: laziness percentage
    :param i_gi: grid interval percentage
    :param i_grids: number of grids
    :param smooth_type: type of smoothing function to use
    :param kwargs: additional parameters for the smoothing function
    :return: buy signals, sell signals, grid lines
    """
    i_gi = i_gi / 100
    i_lz = round(i_lz * 4) / 400
    i_cool = max(0, i_cool)

    try:
        smoothing_func = get_smoothing_function(smooth_type)
        ma = smoothing_func(candles['close'], i_len, **kwargs)
    except Exception as e:
        raise ValueError(f"Error in smoothing function: {str(e)}")

    def lz(ma: pd.Series, lzf: float) -> List[float]:
        s = np.sign(ma)
        lz = [ma.iloc[0]]
        for i in range(1, len(ma)):
            if (ma.iloc[i] == ma.iloc[i - 1]) or (ma.iloc[i] > lz[-1] + lzf * lz[-1] * s.iloc[i]) or (
                    ma.iloc[i] < lz[-1] - lzf * lz[-1] * s.iloc[i]):
                lz.append(ma.iloc[i])
            else:
                lz.append(lz[-1])
        return lz

    df = pd.DataFrame()
    df['ma'] = ma.fillna(0)
    df['lma'] = lz(df['ma'], i_lz)

    elstx = mintick * i_elstx

    ap = [0]
    gi = []
    next_up = [0]
    next_down = [0]
    a_grid_cont = []
    upper_limit = []
    lower_limit = []
    buy_indexes = []
    sell_indexes = []
    buy = []
    sell = []
    signal_line = [0]
    last_signal_index = [0]
    last_signal = [0]

    for index, row in df.iterrows():
        # Calculate ap
        if row['ma'] > row['lma']:
            ap_temp = ap[-1] + elstx
        elif row['ma'] < row['lma']:
            ap_temp = ap[-1] - elstx
        else:
            ap_temp = ap[-1]

        ap_temp = max(min(ap_temp, next_up[-1]), next_down[-1])
        ap_temp = row['lma'] if row['lma'] != df['lma'].iloc[index - 1] else ap_temp
        ap.append(ap_temp)

        # Calculate gi
        gi.append(ap_temp * i_gi)

        # Calculate next_up and next_down
        if row['lma'] != df['lma'].iloc[index - 1]:
            next_up.append(row['lma'] + row['lma'] * i_gi)
            next_down.append(row['lma'] - row['lma'] * i_gi)
        else:
            next_up.append(next_up[-1])
            next_down.append(next_down[-1])

        # Calculate grids
        a_grid = [ap_temp + gi[-1] * x for x in range(-4, 5)]
        a_grid_cont.append(a_grid)

        # Calculate upper/lower limits
        if i_grids >= 8:
            upper_limit.append(a_grid[8])
            lower_limit.append(a_grid[0])
        elif i_grids >= 6:
            upper_limit.append(a_grid[7])
            lower_limit.append(a_grid[1])
        elif i_grids >= 4:
            upper_limit.append(a_grid[6])
            lower_limit.append(a_grid[2])
        else:
            upper_limit.append(a_grid[5])
            lower_limit.append(a_grid[3])

        # Create buy and sell signals
        start = int(4 - i_grids / 2)
        end = int(4 + i_grids / 2)
        buy_index = 0
        sell_index = 0
        for x in range(start, end + 1):
            value = a_grid[x]
            if i_extr:
                sell_index = x if candles['low'].iloc[index - 1] < value and candles['high'].iloc[
                    index] >= value else sell_index
                buy_index = x if candles['high'].iloc[index - 1] > value and candles['low'].iloc[
                    index] <= value else buy_index
            else:
                sell_index = x if candles['close'].iloc[index - 1] < value and candles['close'].iloc[
                    index] >= value else sell_index
                buy_index = x if candles['close'].iloc[index - 1] > value and candles['close'].iloc[
                    index] <= value else buy_index
        buy_indexes.append(buy_index)
        sell_indexes.append(sell_index)

        buy_sig = buy_index > 0
        sell_sig = sell_index > 0

        buy_sig = False if candles['low'].iloc[index] >= signal_line[-1] - gi[-1] else buy_sig
        sell_sig = False if candles['high'].iloc[index] <= signal_line[-1] + gi[-1] else sell_sig

        buy_sig = False if candles['close'].iloc[index] > upper_limit[-1] or candles['close'].iloc[index] < lower_limit[
            -1] else buy_sig
        sell_sig = False if candles['close'].iloc[index] < lower_limit[-1] or candles['close'].iloc[index] > \
                            upper_limit[-1] else sell_sig

        direction = 1 if i_dir == 'up' else -1 if i_dir == 'down' else 0
        buy_sig = False if direction == -1 and candles['low'].iloc[index] >= signal_line[-1] - gi[-1] * 2 else buy_sig
        sell_sig = False if direction == 1 and candles['high'].iloc[index] <= signal_line[-1] + gi[-1] * 2 else sell_sig

        if buy_sig and sell_sig:
            buy_sig = sell_sig = False

        y = sum(1 for i in range(1, min(i_cool + 1, len(buy))) if not (buy[-i] or sell[-i]))
        buy_sig = sell_sig = False if y < i_cool else buy_sig

        buy.append(buy_sig)
        sell.append(sell_sig)

        last_signal.append(1 if buy_sig else -1 if sell_sig else last_signal[-1])
        last_signal_index.append(buy_index if buy_sig else sell_index if sell_index else last_signal_index[-1])
        signal_line.append(a_grid[last_signal_index[-1]])

        if i_reset:
            signal_line[-1] = upper_limit[-1] if row['lma'] < df['lma'].iloc[index - 1] else signal_line[-1]
            signal_line[-1] = lower_limit[-1] if row['lma'] > df['lma'].iloc[index - 1] else signal_line[-1]

    return pd.Series(buy), pd.Series(sell), a_grid_cont


def grid_system(
        df: pd.DataFrame,
        grid_type: str = 'auto',
        i_extr: bool = False,
        bars_back: int = 200,
        grid_params: str = 'Chart Range',
        grid_section: int = 1,
        display_levels: int = 5,
        calculation: str = 'arithmetic',
        projection_mode: str = 'basic',
        smooth_type: str = 'ema',
        **kwargs
) -> Tuple[pd.Series, pd.Series, List[List[float]]]:
    """
    Unified function to calculate grid lines and signals using either FibGrid or AutoGrid.

    :param df: DataFrame with OHLCV data
    :param grid_type: 'fib' for FibGrid, 'auto' for AutoGrid
    :param i_extr: boolean of 'use highs/lows for signals'
    :param bars_back: number of bars to look back for highest/lowest
    :param grid_params: grid parameters preset
    :param grid_section: grid section
    :param display_levels: number of display levels
    :param calculation: calculation mode
    :param projection_mode: projection mode
    :param smooth_type: type of smoothing function to use
    :param kwargs: additional parameters for grid functions
    :return: buy signals, sell signals, grid lines
    """
    if grid_type.lower() == 'fib':
        fire, tree, snow, sky, dm = fibgrid_lines(
            df,
            bars_back=bars_back,
            grid_params=grid_params,
            grid_section=grid_section,
            display_levels=display_levels,
            calculation=calculation,
            projection_mode=projection_mode,
            **kwargs
        )

        grid_lines = sorted([ele for ele in fire + tree + snow + sky + dm if ele != 0])

        buys = []
        sells = []
        for index, row in df.iterrows():
            buy = sell = False
            for value in grid_lines:
                if i_extr:
                    if df['low'].iloc[index - 1] < value <= row['high']:
                        sell = True
                        break
                    if df['high'].iloc[index - 1] > value >= row['low']:
                        buy = True
                        break
                else:
                    if df['close'].iloc[index - 1] < value <= row['close']:
                        sell = True
                        break
                    if df['close'].iloc[index - 1] > value >= row['close']:
                        buy = True
                        break
            buys.append(buy)
            sells.append(sell)

        return pd.Series(buys), pd.Series(sells), [fire, tree, snow, sky, dm]

    elif grid_type.lower() == 'auto':
        return auto_grid(df, smooth_type=smooth_type, **kwargs)

    else:
        raise ValueError("Invalid grid_type. Choose 'fib' or 'auto'.")


class GridSystemConfig(DirectionalTradingControllerConfigBase):
    controller_name = "grid_system"
    candles_config: List[CandlesConfig] = []

    # Common parameters
    connector: str = Field(default="kucoin", client_data=ClientFieldData(prompt=lambda msg: "Enter the exchange connector", prompt_on_new=True))
    trading_pair: str = Field(default="BTC-USDT", client_data=ClientFieldData(prompt=lambda msg: "Enter the trading pair", prompt_on_new=True))
    grid_type: str = Field(default="auto", client_data=ClientFieldData(prompt=lambda msg: "Select grid type (auto, fib)", prompt_on_new=True))
    smooth_type: str = Field(default="ema", client_data=ClientFieldData(prompt=lambda msg: f"Select smoothing function {list(available_smoothing_functions().keys())}", prompt_on_new=True))
    length: int = Field(default=100, client_data=ClientFieldData(prompt=lambda msg: "Enter the grid length", prompt_on_new=True))
    bars_back: int = Field(default=200, client_data=ClientFieldData(prompt=lambda msg: "Enter the number of bars to look back", prompt_on_new=True))
    i_extr: bool = Field(default=True, client_data=ClientFieldData(prompt=lambda msg: "Use highs/lows for signals? (True/False)", prompt_on_new=True))

    # Fibgrid parameters
    grid_params: str = Field(default="Chart Range", client_data=ClientFieldData(prompt=lambda msg: "Enter grid parameters preset", prompt_on_new=True))
    grid_section: int = Field(default=5, client_data=ClientFieldData(prompt=lambda msg: "Enter grid section", prompt_on_new=True))
    subsection: str = Field(default="none", client_data=ClientFieldData(prompt=lambda msg: "Enter subsection"))
    calculation: str = Field(default="arithmetic", client_data=ClientFieldData(prompt=lambda msg: "Enter calculation mode (arithmetic/geometric)", prompt_on_new=True))

    projection_mode: str = Field(default="basic", client_data=ClientFieldData(prompt=lambda msg: "Enter projection mode (basic/high/fixed point/61.8 level)", prompt_on_new=True))
    display_levels: int = Field(default=2, client_data=ClientFieldData(prompt=lambda msg: "Enter number of display levels", prompt_on_new=True))
    start: Decimal = Field(default=Decimal("63.00"), client_data=ClientFieldData(prompt=lambda msg: "Enter start value for manual grid", prompt_on_new=True))
    stop: Decimal = Field(default=Decimal("1163.00"), client_data=ClientFieldData(prompt=lambda msg: "Enter stop value for manual grid", prompt_on_new=True))
    projection_point: Decimal = Field(default=Decimal("0.8346"), client_data=ClientFieldData(prompt=lambda msg: "Enter projection point for 'fixed point' mode", prompt_on_new=True))

    # Auto_grid parameters
    i_len: int = Field(default=7, client_data=ClientFieldData(prompt=lambda msg: "Enter smoothing length for auto_grid", prompt_on_new=True))
    i_dir: str = Field(default="neutral", client_data=ClientFieldData(prompt=lambda msg: "Enter direction for auto_grid (neutral/up/down)", prompt_on_new=True))
    i_cool: int = Field(default=2, client_data=ClientFieldData(prompt=lambda msg: "Enter cooldown period for auto_grid", prompt_on_new=True))
    i_reset: bool = Field(default=True, client_data=ClientFieldData(prompt=lambda msg: "Reset buy/sell index when grids change? (True/False)", prompt_on_new=True))
    mintick: Decimal = Field(default=Decimal("1"), client_data=ClientFieldData(prompt=lambda msg: "Enter minimum tick amount for the symbol", prompt_on_new=True))
    i_elstx: int = Field(default=50, client_data=ClientFieldData(prompt=lambda msg: "Enter elasticity for auto_grid",
                                                                 prompt_on_new=True))
    i_lz: int = Field(default=4, client_data=ClientFieldData(prompt=lambda msg: "Enter laziness percentage for auto_grid", prompt_on_new=True))
    i_gi: int = Field(default=2, client_data=ClientFieldData(prompt=lambda msg: "Enter grid interval percentage for auto_grid", prompt_on_new=True))
    interval: str = Field(default="5m", client_data=ClientFieldData(prompt=lambda msg: "Enter Interval(1m, 1hr, 1d)", prompt_on_new=True))
    i_grids: int = Field(default=6, client_data=ClientFieldData(prompt=lambda msg: "Enter number of grids for auto_grid", prompt_on_new=True))

    # Additional strategy parameters
    order_amount: Decimal = Field(default=Decimal("0.01"), client_data=ClientFieldData(prompt=lambda msg: "Enter the base order amount", prompt_on_new=True))
    leverage: int = Field(default=1, client_data=ClientFieldData(prompt=lambda msg: "Enter the leverage to use for trading", prompt_on_new=True))



class GridSystem(DirectionalTradingControllerBase):
    def __init__(self, config: GridSystemConfig, *args, **kwargs):
        self.config = config
        self.df = pd.DataFrame()
        self.max_records = config.length
        if len(self.config.candles_config) == 0:
            self.config.candles_config = [CandlesConfig(
                connector=config.connector,
                trading_pair=config.trading_pair,
                interval=config.interval,
                max_records=config.bars_back
            )]
        super().__init__(config, *args, **kwargs)

    async def update_processed_data(self):
        candles = await self.market_data_provider.get_candles_df(
            connector_name=self.config.connector,
            trading_pair=self.config.trading_pair,
            interval=self.config.interval,
            max_records=self.config.bars_back
        )
        self.df = candles.candles_df

    async def update_market_data(self):
        candles = await self.market_data_provider.get_candles_df(
            connector_name=self.config.connector,
            trading_pair=self.config.trading_pair,
            interval=self.config.interval,
            max_records=self.config.bars_back
        )
        self.df = candles.candles_df

    async def process_tick(self) -> List[Union[CreateExecutorAction, StopExecutorAction]]:
        await self.update_market_data()
        signals = self.calculate_signals()
        return await self.process_signals(signals)

    def calculate_signals(self):
        # TODO: FIX THE CALCULATION OF THE GRID LINES
        if self.config.grid_type == 'fib':
            return fibgrid_lines(self.df, self.config)
        elif self.config.grid_type == 'auto':
            return auto_grid(self.df, self.config)

    async def process_signals(self, signals) -> List[Union[CreateExecutorAction, StopExecutorAction]]:
        grid_buys, grid_sells, grid_lines = signals
        actions = []

        for i in range(len(grid_buys)):
            if grid_buys.iloc[i]:
                side = TradeType.BUY
                actions.append(CreateExecutorAction(
                    controller_id=self.config.id,
                    executor_config=PositionExecutorConfig(
                        timestamp=self.market_data_provider.time(),
                        trading_pair=self.config.trading_pair,
                        connector_name=self.config.connector,
                        side=side,
                        leverage=self.config.leverage,
                        triple_barrier_config=TripleBarrierConfig(),
                        amount=self.config.order_amount,
                        entry_price=Decimal("11111111")  ##### SELECT THE RIGHT PRICE
                    )))
            elif grid_sells.iloc[i]:
                side = TradeType.SELL
                actions.append(CreateExecutorAction(
                    controller_id=self.config.id,
                    executor_config=PositionExecutorConfig(
                        timestamp=self.market_data_provider.time(),
                        trading_pair=self.config.trading_pair,
                        connector_name=self.config.connector,
                        side=side,
                        leverage=self.config.leverage,
                        triple_barrier_config=TripleBarrierConfig(),
                        amount=self.config.order_amount,
                        entry_price=Decimal("11111111")  ##### SELECT THE RIGHT PRICE
                    )
                ))

        return actions

    def to_format_status(self) -> List[str]:
        return [
            f"Grid Type: {self.config.grid_type}",
            f"Smoothing Type: {self.config.smooth_type}",
            f"Bars Back: {self.config.bars_back}",
            f"Order Amount: {self.config.order_amount}",
            f"Leverage: {self.config.leverage}",
        ]
