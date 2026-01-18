from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd


REQUIRED_COLS = ("Date", "Open", "High", "Low", "Close")
OPTIONAL_COLS = ("Volume",)


@dataclass(frozen=True)
class DataQualityReport:
    start: pd.Timestamp
    end: pd.Timestamp
    timeframe: str
    n_rows_raw: int
    n_rows_clean: int
    n_duplicates_dropped: int
    n_invalid_dates_dropped: int
    n_invalid_prices_dropped: int
    n_missing_close: int
    calendar: str
    expected_bars: int
    missing_bars: int
    filled_bars: int
    left_missing_bars: int


# load data


def load_xau_15m_csv(
    path: str,
    sep: Optional[str] = None,
    encoding: str = "utf-8-sig",
) -> pd.DataFrame:
    if sep is None:
        df = pd.read_csv(path, sep=None, engine="python", encoding=encoding)
    else:
        df = pd.read_csv(path, sep=sep, encoding=encoding)

    if df.empty:
        raise ValueError(f"CSV loaded but is empty: {path}")

    return df


# validacija ir valymas


def _strip_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out


def _require_columns(df: pd.DataFrame, required: Sequence[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found: {list(df.columns)}")


def clean_xau_15m(
    df: pd.DataFrame,
    date_format: str = "%Y.%m.%d %H:%M",
    tz: Optional[str] = None,
) -> Tuple[pd.DataFrame, dict]:
    stats = {
        "n_rows_raw": int(len(df)),
        "n_duplicates_dropped": 0,
        "n_invalid_dates_dropped": 0,
        "n_invalid_prices_dropped": 0,
    }

    out = _strip_columns(df)
    _require_columns(out, REQUIRED_COLS)

    dt = pd.to_datetime(out["Date"], format=date_format, errors="coerce")
    bad_dt = int(dt.isna().sum())
    stats["n_invalid_dates_dropped"] = bad_dt

    out = out.loc[~dt.isna()].copy()
    out["Date"] = dt.loc[~dt.isna()].values

    if tz is not None:
        out["Date"] = out["Date"].dt.tz_localize(
            tz, ambiguous="infer", nonexistent="shift_forward"
        )

    for c in ("Open", "High", "Low", "Close"):
        out[c] = pd.to_numeric(out[c], errors="coerce")

    if "Volume" in out.columns:
        out["Volume"] = pd.to_numeric(out["Volume"], errors="coerce")


    bad_price = int(out["Close"].isna().sum())
    stats["n_invalid_prices_dropped"] = bad_price
    out = out.dropna(subset=["Close"]).copy()

    out = out.sort_values("Date")
    dup = int(out["Date"].duplicated(keep="last").sum())
    stats["n_duplicates_dropped"] = dup
    out = out.drop_duplicates(subset=["Date"], keep="last")

    out = out.set_index("Date").sort_index()

    if not out.index.is_monotonic_increasing:
        raise ValueError("Index must be monotonic increasing after cleaning (unexpected).")

    return out, stats

# weekdays

def build_time_index(
    start: pd.Timestamp,
    end: pd.Timestamp,
    freq: str = "15min",
    calendar: str = "weekday",
) -> pd.DatetimeIndex:
    if start >= end:
        raise ValueError("start must be < end")

    full = pd.date_range(start=start, end=end, freq=freq)
    if calendar == "weekday":
        full = full[full.weekday < 5]
    elif calendar == "all":
        pass
    else:
        raise ValueError("calendar must be 'weekday' or 'all'")
    return full


# filling gaps

def reindex_and_fill_gaps(
    df: pd.DataFrame,
    freq: str = "15min",
    calendar: str = "weekday",
    max_fill_bars: int = 8,
    fill_policy: str = "ffill_close_only",
) -> Tuple[pd.DataFrame, int, int, int]:
    if "Close" not in df.columns:
        raise ValueError("df must contain 'Close' column")

    df = df.copy().sort_index()

    idx = build_time_index(df.index.min(), df.index.max(), freq=freq, calendar=calendar)
    expected = int(len(idx))

    out = df.reindex(idx)
    miss = out["Close"].isna().to_numpy()
    missing = int(miss.sum())

    if fill_policy == "none" or missing == 0:
        return out, expected, missing, 0

    if fill_policy != "ffill_close_only":
        raise ValueError("fill_policy must be 'none' or 'ffill_close_only'")

    m = miss.astype(np.int8)
    dm = np.diff(m, prepend=0, append=0)
    run_starts = np.where(dm == 1)[0]
    run_ends = np.where(dm == -1)[0]
    run_lengths = run_ends - run_starts

    fill_mask = np.zeros(len(out), dtype=bool)
    for s, e, L in zip(run_starts, run_ends, run_lengths):
        if L <= max_fill_bars:
            fill_mask[s:e] = True

    filled = int(fill_mask.sum())

    close_ff = out["Close"].ffill()

    out2 = out.copy()
    fill_idx = out2.index[fill_mask]

    out2.loc[fill_idx, "Close"] = close_ff.loc[fill_idx]

    for c in ("Open", "High", "Low"):
        if c in out2.columns:
            out2.loc[fill_idx, c] = out2.loc[fill_idx, "Close"]

    if "Volume" in out2.columns:
        out2.loc[fill_idx, "Volume"] = 0.0

    if not out["Close"].isna().loc[fill_idx].all():
        raise RuntimeError("Internal error: attempted to fill bars that were not missing.")

    return out2, expected, missing, filled

# returns

def compute_log_returns(
    df: pd.DataFrame,
    price_col: str = "Close",
    scale: float = 100.0,
    dropna: bool = True,
) -> pd.Series:
    if price_col not in df.columns:
        raise ValueError(f"Missing price column: {price_col}")

    close = df[price_col].astype(float)
    r = np.log(close / close.shift(1)) * float(scale)
    r = r.replace([np.inf, -np.inf], np.nan)
    r.name = "log_return"
    return r.dropna() if dropna else r

# data report

def make_data_quality_report(
    df_raw: pd.DataFrame,
    df_clean: pd.DataFrame,
    clean_stats: dict,
    expected_bars: int,
    missing_bars: int,
    filled_bars: int,
    calendar: str,
    freq: str,
) -> DataQualityReport:
    left_missing = int(missing_bars - filled_bars)
    return DataQualityReport(
        start=df_clean.index.min(),
        end=df_clean.index.max(),
        timeframe=freq,
        n_rows_raw=int(len(df_raw)),
        n_rows_clean=int(len(df_clean)),
        n_duplicates_dropped=int(clean_stats.get("n_duplicates_dropped", 0)),
        n_invalid_dates_dropped=int(clean_stats.get("n_invalid_dates_dropped", 0)),
        n_invalid_prices_dropped=int(clean_stats.get("n_invalid_prices_dropped", 0)),
        n_missing_close=int(df_clean["Close"].isna().sum()) if "Close" in df_clean.columns else 0,
        calendar=str(calendar),
        expected_bars=int(expected_bars),
        missing_bars=int(missing_bars),
        filled_bars=int(filled_bars),
        left_missing_bars=int(left_missing),
    )


def print_data_quality_report(rep: DataQualityReport) -> None:
    print("\n=== DATA QUALITY REPORT ===")
    print(f"Timeframe: {rep.timeframe}")
    print(f"Calendar: {rep.calendar}")
    print(f"Coverage: {rep.start} → {rep.end}")
    print(f"Raw rows: {rep.n_rows_raw}")
    print(f"Clean rows: {rep.n_rows_clean}")
    print(f"Dropped invalid dates: {rep.n_invalid_dates_dropped}")
    print(f"Dropped invalid prices: {rep.n_invalid_prices_dropped}")
    print(f"Dropped duplicates: {rep.n_duplicates_dropped}")
    print(f"Expected bars: {rep.expected_bars}")
    print(f"Missing bars: {rep.missing_bars}")
    print(f"Filled bars (short gaps): {rep.filled_bars}")
    print(f"Left missing (long gaps): {rep.left_missing_bars}")
