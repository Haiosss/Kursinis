from __future__ import annotations

from typing import Iterable, List, Optional

import pandas as pd

from strategy import build_signals, generate_position_from_signals
from metrics import evaluate_performance
from simulation_eval import roundtrip_bps_to_cost_per_unit_change


PERIODS_PER_YEAR_15M = 96 * 252


def optimize_ema_parameters(
    df_in: pd.DataFrame,
    fast_range: Iterable[int],
    slow_range: Iterable[int],
    trend_period: int = 200,
    lag: int = 1,
    periods_per_year: int = PERIODS_PER_YEAR_15M,
    round_trip_bps: float = 10.0,
    out_csv_path: Optional[str] = None,
) -> pd.DataFrame:
    if "Close" not in df_in.columns:
        raise ValueError("df_in must contain a 'Close' column")

    close = df_in["Close"].astype(float)

    cost_per_unit_change = roundtrip_bps_to_cost_per_unit_change(round_trip_bps)

    results: List[dict] = []

    for fast in fast_range:
        for slow in slow_range:
            if fast >= slow:
                continue

            sig = build_signals(
                df_in,
                fast_period=fast,
                slow_period=slow,
                trend_period=trend_period,
                price_col="Close",
            )
            pos = generate_position_from_signals(sig)

            m, ts = evaluate_performance(
                close=close,
                position=pos,
                periods_per_year=periods_per_year,
                lag=lag,
                cost_per_unit_change=cost_per_unit_change,
                start_equity=1.0,
            )

            results.append({
                "fast_period": int(fast),
                "slow_period": int(slow),

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
                "trend_period": int(trend_period),
                "lag": int(lag),
            })

    df_out = pd.DataFrame(results)
    
    # risk and activity filters
    
    filtered = df_out[
        (df_out["max_drawdown"] >= -0.30) &
        (df_out["n_round_trips"] >= 200)
    ].copy()

    if filtered.empty:
        filtered = df_out

    df_out = filtered.sort_values(
        by=[
            "calmar",
            "sortino",
            "sharpe",
            "cagr",
            "turnover",
        ],
        ascending=[False, False, False, False, True],
    ).reset_index(drop=True)


    if out_csv_path is not None:
        df_out.to_csv(out_csv_path, index=False)

    return df_out
