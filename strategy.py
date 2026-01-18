from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

# indikatoriai

def ema(series: pd.Series, span: int) -> pd.Series:
    if span < 1:
        raise ValueError("EMA span must be >= 1")
    return series.ewm(span=int(span), adjust=False).mean()


def crossover(a: pd.Series, b: pd.Series) -> pd.Series:
    x = (a > b) & (a.shift(1) <= b.shift(1))
    return x.fillna(False)


def crossunder(a: pd.Series, b: pd.Series) -> pd.Series:
    x = (a < b) & (a.shift(1) >= b.shift(1))
    return x.fillna(False)


# signals


@dataclass(frozen=True)
class SignalPack:
    ema_fast: pd.Series
    ema_slow: pd.Series
    ema_trend: pd.Series
    long_entry: pd.Series
    short_entry: pd.Series
    long_exit: pd.Series
    short_exit: pd.Series


def build_signals(
    df: pd.DataFrame,
    fast_period: int,
    slow_period: int,
    trend_period: int = 200,
    price_col: str = "Close",
) -> SignalPack:
    if price_col not in df.columns:
        raise ValueError(f"Missing price column: {price_col}")
    if fast_period < 1 or slow_period < 1 or trend_period < 1:
        raise ValueError("All EMA periods must be >= 1")
    if fast_period >= slow_period:
        raise ValueError("fast_period must be < slow_period")

    if not df.index.is_monotonic_increasing:
        df = df.sort_index()

    price = df[price_col].astype(float)

    ema_fast = ema(price, fast_period)
    ema_slow = ema(price, slow_period)
    ema_trend = ema(price, trend_period)

    trend_long_ok = (price > ema_trend).fillna(False)
    trend_short_ok = (price < ema_trend).fillna(False)

    bull_cross = crossover(ema_fast, ema_slow)
    bear_cross = crossunder(ema_fast, ema_slow)

    long_entry = (bull_cross & trend_long_ok).astype(bool)
    short_entry = (bear_cross & trend_short_ok).astype(bool)

    long_exit = (bear_cross | (~trend_long_ok)).astype(bool)
    short_exit = (bull_cross | (~trend_short_ok)).astype(bool)

    return SignalPack(
        ema_fast=ema_fast,
        ema_slow=ema_slow,
        ema_trend=ema_trend,
        long_entry=long_entry,
        short_entry=short_entry,
        long_exit=long_exit,
        short_exit=short_exit,
    )

# position generation

def generate_position_from_signals(
    signals: SignalPack,
    index: Optional[pd.Index] = None,
) -> pd.Series:
    if index is None:
        index = signals.long_entry.index

    le = signals.long_entry.to_numpy(dtype=bool)
    se = signals.short_entry.to_numpy(dtype=bool)
    lx = signals.long_exit.to_numpy(dtype=bool)
    sx = signals.short_exit.to_numpy(dtype=bool)

    n = len(le)
    pos = np.zeros(n, dtype=np.int8)

    current = np.int8(0)
    for i in range(n):
        if le[i]:
            current = np.int8(1)
        elif se[i]:
            current = np.int8(-1)
        else:
            if current == 1 and lx[i]:
                current = np.int8(0)
            elif current == -1 and sx[i]:
                current = np.int8(0)

        pos[i] = current

    return pd.Series(pos, index=index, name="position")


def generate_position(
    df: pd.DataFrame,
    fast_period: int,
    slow_period: int,
    trend_period: int = 200,
    price_col: str = "Close",
) -> pd.Series:
    sig = build_signals(
        df=df,
        fast_period=fast_period,
        slow_period=slow_period,
        trend_period=trend_period,
        price_col=price_col,
    )
    return generate_position_from_signals(sig)
