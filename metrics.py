from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TradeStats:
    n_entries: int
    n_exits: int
    n_flips: int
    n_round_trips: int
    avg_hold_bars: float
    median_hold_bars: float
    turnover: float

# simple returns

def simple_returns(close: pd.Series) -> pd.Series:
    c = close.astype(float)
    r = c.pct_change(fill_method=None)
    r = r.replace([np.inf, -np.inf], np.nan)
    r.name = "simple_return"
    return r

# positions and costs

def align_position(position: pd.Series, index: pd.Index, lag: int = 1) -> pd.Series:
    if lag < 0:
        raise ValueError("lag must be >= 0")
    pos = position.shift(lag).reindex(index)
    pos = pos.fillna(0.0).astype(float)
    pos.name = "position_exec"
    return pos


def transaction_costs_from_position(position_exec: pd.Series, cost_per_unit_change: float) -> pd.Series:
    pos = position_exec.astype(float)
    dpos = pos.diff().abs().fillna(0.0)
    costs = dpos * float(cost_per_unit_change)
    costs.name = "tx_cost"
    return costs


def apply_position_to_returns(
    returns: pd.Series,
    position: pd.Series,
    lag: int = 1,
    cost_per_unit_change: float = 0.0,
) -> Tuple[pd.Series, pd.Series]:
    r = returns.astype(float)
    pos_exec = align_position(position, r.index, lag=lag)

    gross = pos_exec * r
    gross.name = "gross_strategy_return"

    if cost_per_unit_change and cost_per_unit_change != 0.0:
        costs = transaction_costs_from_position(pos_exec, cost_per_unit_change)
        net = gross - costs
    else:
        net = gross

    net.name = "strategy_return"
    return net, pos_exec

# equity and drawdown

def equity_curve_from_simple_returns(strategy_returns: pd.Series, start_equity: float = 1.0) -> pd.Series:
    r = strategy_returns.dropna().astype(float)
    eq = float(start_equity) * (1.0 + r).cumprod()
    eq.name = "equity"
    return eq


def drawdown_series(equity: pd.Series) -> pd.Series:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    dd.name = "drawdown"
    return dd


def max_drawdown(equity: pd.Series) -> float:
    return float(drawdown_series(equity).min())

# summary metrics

def annualized_volatility(returns: pd.Series, periods_per_year: int) -> float:
    r = returns.dropna().astype(float)
    if r.size < 2:
        return 0.0
    return float(r.std(ddof=0) * np.sqrt(float(periods_per_year)))

def sharpe_ratio(returns: pd.Series, periods_per_year: int, rf_per_period: float = 0.0) -> float:
    r = returns.dropna().astype(float)
    if r.size < 2:
        return 0.0
    excess = r - float(rf_per_period)
    sd = excess.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return 0.0
    return float(excess.mean() / sd * np.sqrt(float(periods_per_year)))

def sortino_ratio(returns: pd.Series, periods_per_year: int, mar_per_period: float = 0.0) -> float:
    r = returns.dropna().astype(float)
    if r.size < 2:
        return 0.0

    downside = (r - float(mar_per_period))
    downside = downside[downside < 0.0]

    if downside.size < 2:
        return 0.0

    downside_std = float(downside.std(ddof=0))
    if downside_std == 0.0 or np.isnan(downside_std):
        return 0.0

    mean_excess = float((r - float(mar_per_period)).mean())
    return float(mean_excess / downside_std * np.sqrt(float(periods_per_year)))


def calmar_ratio(equity: pd.Series, periods_per_year: int) -> float:
    e = equity.dropna().astype(float)
    if e.size < 2:
        return 0.0

    dd = abs(float(max_drawdown(e)))
    if dd == 0.0 or np.isnan(dd):
        return 0.0

    cagr = float(cagr_from_equity(e, periods_per_year))
    return float(cagr / dd)


def cagr_from_equity(equity: pd.Series, periods_per_year: int) -> float:
    e = equity.dropna().astype(float)
    if e.size < 2:
        return 0.0
    years = (e.size - 1) / float(periods_per_year)
    if years <= 0:
        return 0.0
    return float(e.iloc[-1] ** (1.0 / years) - 1.0)


def total_return_from_equity(equity: pd.Series) -> float:
    e = equity.dropna().astype(float)
    if e.empty:
        return 0.0
    return float(e.iloc[-1] - 1.0)

# trading diagnostics

def trade_stats_from_position(position_exec: pd.Series) -> TradeStats:
    pos = position_exec.fillna(0.0).astype(float)
    prev = pos.shift(1).fillna(0.0)
    curr = pos

    entries = int(((prev == 0.0) & (curr != 0.0)).sum())
    exits = int(((prev != 0.0) & (curr == 0.0)).sum())
    flips = int(((prev != 0.0) & (curr != 0.0) & (np.sign(prev) != np.sign(curr))).sum())

    turnover = float(pos.diff().abs().fillna(0.0).mean())

    in_pos = (pos != 0.0).to_numpy(dtype=bool)
    m = in_pos.astype(np.int8)
    dm = np.diff(m, prepend=0, append=0)
    starts = np.where(dm == 1)[0]
    ends = np.where(dm == -1)[0]
    lengths = ends - starts

    avg_hold = float(np.mean(lengths)) if lengths.size else 0.0
    med_hold = float(np.median(lengths)) if lengths.size else 0.0

    return TradeStats(
        n_entries=entries,
        n_exits=exits,
        n_flips=flips,
        n_round_trips=exits,
        avg_hold_bars=avg_hold,
        median_hold_bars=med_hold,
        turnover=turnover,
    )

# evaluation

def evaluate_performance(
    close: pd.Series,
    position: pd.Series,
    periods_per_year: int,
    lag: int = 1,
    cost_per_unit_change: float = 0.0,
    start_equity: float = 1.0,
) -> Tuple[Dict[str, float], TradeStats]:
    r = simple_returns(close.astype(float))
    strat_r, pos_exec = apply_position_to_returns(
        returns=r,
        position=position,
        lag=lag,
        cost_per_unit_change=cost_per_unit_change,
    )

    eq = equity_curve_from_simple_returns(strat_r, start_equity=start_equity)

    metrics = {
        "total_return": total_return_from_equity(eq),
        "cagr": cagr_from_equity(eq, periods_per_year),
        "max_drawdown": max_drawdown(eq),
        "ann_vol": annualized_volatility(strat_r, periods_per_year),
        "sharpe": sharpe_ratio(strat_r, periods_per_year),
        "sortino": sortino_ratio(strat_r, periods_per_year, mar_per_period=0.0),
        "calmar": calmar_ratio(eq, periods_per_year),
    }

    ts = trade_stats_from_position(pos_exec)
    return metrics, ts
