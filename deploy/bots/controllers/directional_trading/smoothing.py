from typing import Callable, Dict, Iterable, Tuple
import pandas as pd
import pandas_ta as ta
from math import *
import numpy as np

def sma(src, length):
    return ta.sma(src, length)

def ema(src, length):
    return ta.ema(src, length)

def wma(src, length):
    return ta.wma(src, length)

def dema(src, length):
    ema1 = ta.ema(src, length)
    ema2 = ta.ema(ema1, length)
    return (2 * ema1) - ema2

def tema(src, length):
    ema1 = ta.ema(src, length)
    ema2 = ta.ema(ema1, length)
    ema3 = ta.ema(ema2, length)
    return (3 * ema1) - (3 * ema2) + ema3

def tma(src, length):
    return sma(sma(src, ceil(length/2)), (length//2) + 1)

def vwma(src, volume, length):
    return ta.vwma(src, volume, length)

def hma(src, length):
    return wma(2 * wma(src, length//2) - wma(src, length), int(sqrt(length)))

def jurik(src, length, phase=0.5, power=2):
    r1 = 0.5 if phase < -100 else 2.5 if phase > 100 else phase/100 + 1.5
    beta = 0.45 * (length-1) / (0.45 * (length-1) + 2)
    alpha = beta**power

    e0 = [0]
    det0 = [0]
    e1 = []
    det1 = [0]
    jma = [0]
    for index, val in enumerate(src.tolist()):
        e0.append((1-alpha) * val + alpha * e0[-1])
        det0.append((val - e0[-1]) * (1-beta) + beta*det0[-1])
        e1.append(e0[-1] + r1*det0[-1])
        det1.append((e1[-1] - jma[-1]) * (1-alpha)**2 + alpha**2 * det1[-1])
        jma.append(det1[-1] + jma[-1])

    return pd.Series(jma[1:])

def kama(src, length):
    change = abs(src - src.shift(1))
    volatility = change.rolling(window=length).sum()
    efficiency_ratio = (abs(src - src.shift(length)) / volatility).fillna(0).clip(0, 1)
    smoothing_constant = (efficiency_ratio * (2.0 / (2 + 1) - 2.0 / (30 + 1)) + 2.0 / (30 + 1)) ** 2

    kama = [src.iloc[0]]
    for i in range(1, len(src)):
        kama.append(kama[-1] + smoothing_constant.iloc[i] * (src.iloc[i] - kama[-1]))
    return pd.Series(kama, index=src.index)

def zlema(src, length):
    lag = (length - 1) // 2
    return ema(src + (src - src.shift(lag)), length)

def t3(src, length, vfactor=0.7):
    e1 = ema(src, length)
    e2 = ema(e1, length)
    e3 = ema(e2, length)
    e4 = ema(e3, length)
    e5 = ema(e4, length)
    e6 = ema(e5, length)
    
    c1 = -vfactor * vfactor * vfactor
    c2 = 3 * vfactor * vfactor + 3 * vfactor * vfactor * vfactor
    c3 = -6 * vfactor * vfactor - 3 * vfactor - 3 * vfactor * vfactor * vfactor
    c4 = 1 + 3 * vfactor + vfactor * vfactor * vfactor + 3 * vfactor * vfactor
    
    return c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3

def rma(src, length):
    return ta.rma(src, length)

def dv2(df, length):
    dv2 = (df['open'].shift(length) + df['close']) / 2
    return dv2

def modFilt(src, length, z=0.5, beta=0.8):
    z = 0 if z < 0 else 1 if z > 1 else z
    alpha = 2/(length+1)
    beta = 0 if beta < 0 else 1 if beta > 1 else beta

    a = []
    b = []
    c = []
    upper = []
    lower = []
    ts = [0]
    os = [0]
    for index, val in enumerate(src.tolist()):
        if index == 0:
            a.append(z*val + (1-z)*val)
            if a[-1] > alpha*a[-1] + (1-alpha)*a[-1]:
                b.append(a[-1])
            else:
                b.append(alpha*a[-1] + (1-alpha)*a[-1])

            if a[-1] < alpha*a[-1] + (1-alpha)*a[-1]:
                c.append(a[-1])
            else:
                c.append(alpha*a[-1] + (1-alpha)*a[-1])
        else:
            a.append(z*val + (1-z)*ts[-1])
            if a[-1] > alpha*a[-1]+(1-alpha)*b[-1]:
                b.append(a[-1])
            else:
                b.append(alpha*a[-1]+(1-alpha)*b[-1])

            if a[-1] < alpha*a[-1]+(1-alpha)*c[-1]:
                c.append(a[-1])
            else:
                c.append(alpha*a[-1]+(1-alpha)*c[-1])

        if a[-1] == b[-1]:
            os.append(1)
        elif a[-1] == c[-1]:
            os.append(0)
        else:
            os.append(os[-1])

        upper.append(beta*b[-1]+(1-beta)*c[-1])
        lower.append(beta*c[-1]+(1-beta)*b[-1])
        ts.append(os[-1]*upper[-1]+(1-os[-1])*lower[-1])

    return pd.Series(ts)

def smma(src, length):
    sum1 = src.rolling(window=length).sum()
    smma1 = src.rolling(window=length).sum()/length
    return (sum1 - smma1 + src) / length

def covwma(src, length):
    def stdev(close, length):
        return ta.stdev(close, length)
    
    sma1 = sma(src, length)
    dev = stdev(src, length)
    cv = dev/sma1
    cw = src*cv

    return cw.rolling(window=length).sum() / cv.rolling(window=length).sum()

def frama(df, length):
    coefficient = -4.6
    df['n3'] = (df['high'].rolling(window=length).max() - df['low'].rolling(window=length).min()) / length
    df['hd2'] = df['high'].rolling(window=length//2).max()
    df['ld2'] = df['low'].rolling(window=length//2).min()
    df['n2'] = (df['hd2'] - df['ld2']) / (length / 2)
    df['n1'] = (df['hd2'].shift(length//2) - df['ld2'].shift(length//2)) / (length/2)
    df['dim'] = np.where((df['n1'] > 0) & (df['n2'] > 0) & (df['n3'] > 0), (np.log(df['n1']+df['n2']) - np.log(df['n3'])) / np.log(2), 0)
    df['alpha'] = np.exp(coefficient*(df['dim']-1))
    
    conditions = [
    (df['alpha'] < .01),
    (df['alpha'] > 1),
    (df['alpha'] >= .01) & (df['alpha'] <= 1)]
    choices = [.01, 1, df['alpha']]
    df['sc'] = np.select(conditions, choices, default=1)

    r = df['close']*df['sc'] + df['close'].shift(1)*(1-df['sc'])

    return r

def donchian(df, length):
    return (df['high'].rolling(window=length).max() + df['low'].rolling(window=length).min()) / 2

def vama(src, length):
    mid = ema(src, length)
    dev = src - mid
    vol_up = dev.rolling(window=length).max()
    vol_down = dev.rolling(window=length).min()
    vama = mid + ((vol_up+vol_down) / 2)
    return vama

def mcg(src, length):
    e = ema(src, length)
    mg = [0]*(length)

    for index, val in enumerate(src.tolist()):
        if index < length:
            continue
        elif index == length:
            mg.append(e.iloc[index])
        else:
            mg.append(mg[-1] + (val - mg[-1]) / (length * (val/mg[-1])**4))
    
    return pd.Series(mg)

def xema(src, length):
    mult = 2/(length+1)

    res = []
    for index, val in enumerate(src.tolist()):
        if index == 0:
            res.append(mult * val + (1 - mult) * val)
        else:
            res.append(mult * val + (1 - mult) * res[-1])

    return pd.Series(res)

def EhlersSuperSmoother(src, lower):
    a1 = exp(-pi * sqrt(2) / lower)
    coeff2 = 2 * a1 * cos(sqrt(2) * pi / lower)
    coeff3 = -(a1**2)
    coeff1 = (1 - coeff2 - coeff3) / 2

    filt = [0,0]
    for index, val in enumerate(src.tolist()):
        if index == 0:
            filt.append(coeff1 * (val * 2) + coeff2 * filt[-1] + coeff3 * filt[-2])
        else:
            filt.append(coeff1 * (val + src.tolist()[index-1]) + coeff2 * filt[-1] + coeff3 * filt[-2])

    return pd.Series(filt[2:])

def EhlersEmaSmoother(src, k, p):
    return EhlersSuperSmoother(xema(src, k), p)

def adxvma(src, length):
    diff = (src - src.shift(1)).fillna(0)
    tpdm = np.where(diff > 0, diff, 0)
    tmdm = np.where(diff > 0, 0, -diff)

    tpdm = [0]
    tmdm = [0]
    tpdi = [0]
    tmdi = [0]
    pdm = [0]
    mdm = [0]
    pdi = [0]
    mdi = [0]
    tr = []
    tout = [0]
    out = [0]
    thi = []
    tlo = []
    vi = [0]
    vl = [0]
    for index, val in enumerate(src.tolist()):
        if diff[index] > 0:
            tpdm.append(diff[index])
            tmdm.append(tmdm[-1])
        else:
            tpdm.append(tpdm[-1])
            tmdm.append(-diff[index])

        pdm.append(((length-1) * (pdm[-1] + tpdm[-1])) / length)
        mdm.append(((length-1) * (mdm[-1] + tmdm[-1])) / length)

        tr.append(pdm[-1] + mdm[-1])
        if tr[-1] != 0:
            tpdi.append(pdm[-1]/tr[-1])
            tmdi.append(mdm[-1] / tr[-1])
        else:
            tpdi.append(0)
            tmdi.append(0)

        pdi.append(((length-1) * (pdi[-1] + tpdi[-1])) / length)
        mdi.append(((length-1) * (mdi[-1] + tmdi[-1])) / length)

        if (pdi[-1] + mdi[-1]) > 0:
            tout.append(abs(pdi[-1] - mdi[-1]) / (pdi[-1] + mdi[-1]))
        else:
            tout.append(tout[-1])

        out.append(((length-1) * out[-1] + tout[-1]) / length)
        thi.append(max(out[-length:]))
        tlo.append(min(out[-length:]))

        if (thi[-1] - tlo[-1]) > 0:
            vi.append((out[-1] - tlo[-1]) / (thi[-1] - tlo[-1]))
        else:
            vi.append(vi[-1])

        vl.append(((length-vi[-1]) * vl[-1] + vi[-1] * val) / length)

    return pd.Series(vl[1:])

def ahrma(src, length):
    med = [0]
    ahr = [0]
    for index, val in enumerate(src.tolist()):
        try:
            med.append((ahr[-1] + ahr[-length]) / 2)
        except:
            med.append(0)
        ahr.append(ahr[-1] + ((val - med[-1]) / length))

    return pd.Series(ahr[1:])

def alxma(src, length):
    out = []
    for index, val in enumerate(src.tolist()):
        sumw1 = length - 2
        sums1 = sumw1 * val
        
        for k in range(1, length+1):
            weight = length - k - 2
            sumw1 += weight
            sums1 += weight * src.iloc[index-k]

        out.append(val) if length < 4 else out.append(sums1/sumw1)

    return pd.Series(out)

def ie2(src, length):
    a = .7
    e1 = ema(src, length)
    e2 = ema(e1, length)
    e3 = ema(e2, length)
    e4 = ema(e3, length)
    e5 = ema(e4, length)
    e6 = ema(e5, length)

    out = []
    for index, val in enumerate(src.tolist()):
        c1 = -a**3
        c2 = 3*a**2 + 3*a**3
        c3 = -6*a**2 - 3*a - 3*a**3
        c4 = 1 + 3*a + a**3 + 3*a**2

        out.append(c1*e6.iloc[index] + c2*e5.iloc[index] + c3*e4.iloc[index] + c4*e3.iloc[index])
        
    return pd.Series(out)

def ilrs(src, length):
    sma1 = sma(src, length)
    sums = length * (length-1) * .5
    sum2 = (length-1) * length * (2 * length - 1) / 6

    out = []
    for index, val in enumerate(src.tolist()):
        sum1 = 0
        sumy = 0
        slope = 0

        for i in range(length):
            sum1 += i * src.iloc[index-i]
            sumy += src.iloc[index-i]
        num1 = length * sum1 - sums * sumy
        num2 = sums * sums - length * sum2
        slope = num1/num2 if num2 != 0 else 0
        out.append(slope + sma1.iloc[index])

    return pd.Series(out)

def leader(src, length):
    alpha = 2/(length+1)

    out = []
    ldr = [0]
    ldr2 = [0]
    for index, val in enumerate(src.tolist()):
        ldr.append(ldr[-1] + alpha * (val - ldr[-1]))
        ldr2.append(ldr2[-1] + alpha * (val - ldr[-1] - ldr2[-1]))

        out.append(ldr[-1] + ldr2[-1])

    return pd.Series(out)

def rmta(src, length):
    alpha = 2 / (length+1)

    out = [0]
    b = [0]
    for index, val in enumerate(src.tolist()):
        b.append((1-alpha) * b[-1] + val)

        out.append((1-alpha) * out[-1] + alpha * (val + b[-1] - b[-2]))

    return pd.Series(out[1:])

def decycler(src, length):
    alphaArg = 2 * pi / (length * sqrt(2))
    alpha = (cos(alphaArg) + sin(alphaArg) - 1) / cos(alphaArg) if cos(alphaArg) != 0 else 0

    out = []
    hp = [0,0]
    for index, val in enumerate(src.tolist()):
        hp.append((1 - alpha/2)**2 * (val - 2 * src.iloc[index-1] + src.iloc[index-2]) + 2 * (1 - alpha) * hp[-1] - (1 - alpha)**2 * hp[-2])
        out.append(val - hp[-1])

    return pd.Series(out)

def ave_function(src, length, type, **kwargs):
    out = None
    try:
        exec(f"out = {type}(src, length)")
    except:
        try:
            exec(f"out = {type}(kwargs['df'], length)")
        except:
            if type == 'jurik':
                out = jurik(src, length, kwargs.get('phase', 0.5), kwargs.get('power', 2))
            elif type == 'EhlersEmaSmoother':
                out = EhlersEmaSmoother(src, kwargs['k'], kwargs['p'])
            elif type == 'modFilt':
                out = modFilt(src, length, kwargs.get('z', 0.5), kwargs.get('beta', 0.8))
            elif type == 'vwma':
                out = vwma(src, kwargs['volume'], length)
    return out

def available_smoothing_functions() -> Dict[str, Callable]:
    return {
        'sma': sma,
        'ema': ema,
        'wma': wma,
        'dema': dema,
        'tema': tema,
        'tma': tma,
        'vwma': vwma,
        'hma': hma,
        'jurik': jurik,
        'kama': kama,
        'zlema': zlema,
        't3': t3,
        'rma': rma,
        'smma': smma,
        'hull': hma,  # Alias for hma
        'ehlers': EhlersSuperSmoother,
        'modFilt': modFilt,
        'covwma': covwma,
        'frama': frama,
        'vama': vama,
        'mcginley': mcg,
        'alma': alxma,  # Arnaud Legoux Moving Average
        'evwma': vwma,  # Elastic Volume Weighted Moving Average (same as vwma)
        'hwma': hma,  # Hull Weighted Moving Average (same as hma)
        'jma': jurik,  # Jurik Moving Average (same as jurik)
        'lsma': ilrs,  # Least Squares Moving Average
        'swma': smma,  # Symmetrically Weighted Moving Average (using smma as approximation)
        'wilders': rma,  # Wilder's Smoothing (same as rma)
    }

def all_smoothing_func_names() -> Iterable[str]:
    return [
        'sma', 'ema', 'wma', 'dema', 'tema', 'tma', 'vwma', 'hma',
        'jurik', 'kama', 'zlema', 't3', 'rma', 'smma', 'hull', 'ehlers',
        'modFilt', 'covwma', 'frama', 'vama', 'mcginley', 'alma', 'evwma',
        'hwma', 'jma', 'lsma', 'swma', 'wilders'
    ]


def get_smoothing_function(name: str) -> Callable:
    functions = available_smoothing_functions()
    if name not in functions:
        raise ValueError(f"Unknown smoothing function: {name}")
    return functions[name]
