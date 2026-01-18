from __future__ import annotations

import os
from typing import Dict, Iterable, List, Tuple, Optional

import numpy as np
import pandas as pd

from strategy import build_signals, generate_position_from_signals
from metrics import evaluate_performance

PERIODS_PER_YEAR_15M = 96 * 252

# kastai

def roundtrip_bps_to_cost_per_unit_change(round_trip_bps: float) -> float:
    rt = float(round_trip_bps) / 10_000.0
    return rt / 2.0

# path evaluation

def evaluate_one_path(
    prices: pd.Series,
    fast_period: int,
    slow_period: int,
    trend_period: int = 200,
    lag: int = 1,
    periods_per_year: int = PERIODS_PER_YEAR_15M,
    round_trip_bps: float = 10.0,
    start_equity: float = 1.0,
) -> Dict[str, float]:
    if fast_period >= slow_period:
        raise ValueError("fast_period must be < slow_period")

    df = pd.DataFrame({"Close": prices.astype(float)})

    sig = build_signals(
        df,
        fast_period=fast_period,
        slow_period=slow_period,
        trend_period=trend_period,
        price_col="Close",
    )
    pos = generate_position_from_signals(sig)

    cost_per_unit_change = roundtrip_bps_to_cost_per_unit_change(round_trip_bps)

    m, ts = evaluate_performance(
        close=df["Close"],
        position=pos,
        periods_per_year=periods_per_year,
        lag=lag,
        cost_per_unit_change=cost_per_unit_change,
        start_equity=start_equity,
    )

    return {
        "fast": int(fast_period),
        "slow": int(slow_period),

        "total_return": float(m["total_return"]),
        "cagr": float(m["cagr"]),
        "max_drawdown": float(m["max_drawdown"]),
        "ann_vol": float(m["ann_vol"]),
        "sharpe": float(m["sharpe"]),
        "sortino": float(m["sortino"]),
        "calmar": float(m["calmar"]),

        "n_entries": int(ts.n_entries),
        "n_exits": int(ts.n_exits),
        "n_flips": int(ts.n_flips),
        "n_round_trips": int(ts.n_round_trips),
        "avg_hold_bars": float(ts.avg_hold_bars),
        "median_hold_bars": float(ts.median_hold_bars),
        "turnover": float(ts.turnover),

        "round_trip_bps": float(round_trip_bps),
        "lag": int(lag),
        "trend_period": int(trend_period),
    }

# simulation resuslts for top 10

def save_per_sim_results_finalists(
    prices_sim: np.ndarray,
    index: pd.DatetimeIndex,
    param_list: List[Tuple[int, int]],
    out_csv_path: str,
    trend_period: int = 200,
    lag: int = 1,
    periods_per_year: int = PERIODS_PER_YEAR_15M,
    round_trip_bps: float = 10.0,
    sim_id_offset: int = 0,
    chunk_rows: int = 200_000,
) -> None:
    n_steps, n_sims = prices_sim.shape
    if len(index) != n_steps:
        raise ValueError(f"index length ({len(index)}) must equal n_steps ({n_steps}).")

    os.makedirs(os.path.dirname(out_csv_path) or ".", exist_ok=True)

    header_written = False
    buffer: List[Dict[str, float]] = []

    if os.path.exists(out_csv_path):
        os.remove(out_csv_path)

    for j in range(n_sims):
        sim_id = sim_id_offset + j
        prices = pd.Series(prices_sim[:, j], index=index)

        for (fast, slow) in param_list:
            row = evaluate_one_path(
                prices=prices,
                fast_period=fast,
                slow_period=slow,
                trend_period=trend_period,
                lag=lag,
                periods_per_year=periods_per_year,
                round_trip_bps=round_trip_bps,
                start_equity=1.0,
            )
            row["sim_id"] = int(sim_id)
            buffer.append(row)

        if len(buffer) >= chunk_rows:
            pd.DataFrame(buffer).to_csv(
                out_csv_path,
                index=False,
                mode="a",
                header=not header_written,
            )
            header_written = True
            buffer.clear()

    if buffer:
        pd.DataFrame(buffer).to_csv(
            out_csv_path,
            index=False,
            mode="a",
            header=not header_written,
        )

# per simulation summary

def summarize_mc_results_csv(
    mc_results_csv: str,
    out_summary_csv: Optional[str] = None,
    dd_threshold: float = -0.30,
    tail_q: float = 0.05,
) -> pd.DataFrame:
    df = pd.read_csv(mc_results_csv)

    required = {"fast", "slow", "total_return", "max_drawdown", "sharpe", "sortino", "calmar"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in mc_results_csv: {missing}")

    rows = []
    for (fast, slow), g in df.groupby(["fast", "slow"], sort=False):
        r = g["total_return"].to_numpy(dtype=float)
        dd = g["max_drawdown"].to_numpy(dtype=float)
        sh = g["sharpe"].to_numpy(dtype=float)
        so = g["sortino"].to_numpy(dtype=float)
        ca = g["calmar"].to_numpy(dtype=float)

        p05_r = float(np.quantile(r, tail_q))
        es05 = float(r[r <= p05_r].mean()) if np.any(r <= p05_r) else float("nan")

        rows.append({
            "fast": int(fast),
            "slow": int(slow),

            "mc_mean_return": float(np.mean(r)),
            "mc_median_return": float(np.median(r)),
            "mc_p05_return": p05_r,
            "mc_es05_return": es05,
            "mc_prob_loss": float(np.mean(r < 0.0)),

            "mc_mean_dd": float(np.mean(dd)),
            "mc_p95_dd": float(np.quantile(dd, 0.95)),
            "mc_prob_dd_le_thresh": float(np.mean(dd <= float(dd_threshold))),

            "mc_mean_sharpe": float(np.mean(sh)),
            "mc_p05_sharpe": float(np.quantile(sh, tail_q)),

            "mc_mean_sortino": float(np.mean(so)),
            "mc_p05_sortino": float(np.quantile(so, tail_q)),

            "mc_mean_calmar": float(np.mean(ca)),
            "mc_p05_calmar": float(np.quantile(ca, tail_q)),
        })

    summary = pd.DataFrame(rows)

    # first sort filter

    summary = summary[summary["mc_prob_dd_le_thresh"] <= 0.05].copy()
    if summary.empty:
        summary = pd.DataFrame(rows)

    # second part of sorting

    summary = summary.sort_values(
        by=[
            "mc_p05_return",
            "mc_p05_calmar",
            "mc_p05_sortino",
            "mc_mean_sharpe",
            "mc_prob_loss",
        ],
        ascending=[False, False, False, False, True],
    ).reset_index(drop=True)


    if out_summary_csv is not None:
        os.makedirs(os.path.dirname(out_summary_csv) or ".", exist_ok=True)
        summary.to_csv(out_summary_csv, index=False)

    return summary

# historical grid evaluation

def evaluate_historical_grid_to_csv(
    close: pd.Series,
    index: Optional[pd.DatetimeIndex],
    fast_range: Iterable[int],
    slow_range: Iterable[int],
    out_csv_path: str,
    trend_period: int = 200,
    lag: int = 1,
    periods_per_year: int = PERIODS_PER_YEAR_15M,
    round_trip_bps: float = 10.0,
) -> pd.DataFrame:
    if index is not None:
        prices = pd.Series(close.to_numpy(dtype=float), index=index)
    else:
        prices = close.astype(float)

    os.makedirs(os.path.dirname(out_csv_path) or ".", exist_ok=True)

    rows: List[Dict[str, float]] = []

    for fast in fast_range:
        for slow in slow_range:
            if fast >= slow:
                continue
            row = evaluate_one_path(
                prices=prices,
                fast_period=fast,
                slow_period=slow,
                trend_period=trend_period,
                lag=lag,
                periods_per_year=periods_per_year,
                round_trip_bps=round_trip_bps,
                start_equity=1.0,
            )
            rows.append(row)

    df_out = pd.DataFrame(rows)

    df_sorted = df_out.sort_values(
        by=["sharpe", "total_return", "max_drawdown"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    df_sorted.to_csv(out_csv_path, index=False)
    return df_sorted
