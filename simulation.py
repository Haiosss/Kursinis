from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from arch import arch_model


@dataclass(frozen=True)
class GarchFit:
    model_desc: str
    scale_factor: float
    used_deseasonal: bool
    fit_result: Any 
    returns_used: pd.Series
    intraday_scale: Optional[pd.Series]
    deseasonal_mean: float

# deseasonalisation

def intraday_scaling(returns: pd.Series) -> pd.Series:
    if not isinstance(returns.index, pd.DatetimeIndex):
        raise ValueError("returns must have a DatetimeIndex")

    tod = returns.index.time
    stds = returns.groupby(tod).std()

    mean_std = float(stds.mean())
    if not np.isfinite(mean_std) or mean_std <= 0:
        return pd.Series(1.0, index=stds.index)

    stds = stds.fillna(mean_std)
    scale = (stds / mean_std).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    return scale


def scale_array(index: pd.DatetimeIndex, intraday_scale: pd.Series) -> np.ndarray:
    return np.array([float(intraday_scale.get(ts.time(), 1.0)) for ts in index], dtype=float)


def deseasonalise_returns(returns: pd.Series) -> Tuple[pd.Series, pd.Series]:
    scale = intraday_scaling(returns)
    arr = scale_array(returns.index, scale)
    deseasonal = returns / arr
    deseasonal.name = returns.name
    return deseasonal, scale


# fit GARCH

def fit_garch_11(
    returns: pd.Series,
    mean: str = "Zero",
    dist: str = "normal",
    scale_factor: float = 10.0,
    deseasonalise: bool = True,
    rescale: bool = False,
) -> GarchFit:
    r = returns.dropna().astype(float).copy()
    if r.empty:
        raise ValueError("returns series is empty after dropna()")

    if not isinstance(r.index, pd.DatetimeIndex):
        raise ValueError("returns must have a DatetimeIndex")

    intraday_scale = None
    used_deseasonal = False

    if deseasonalise:
        r_des, intraday_scale = deseasonalise_returns(r)
        r = r_des
        used_deseasonal = True

    deseasonal_mean = float(r.mean())

    r_fit = r * float(scale_factor)

    am = arch_model(
        r_fit,
        mean=mean,
        vol="Garch",
        p=1,
        q=1,
        dist=dist,
        rescale=rescale,
    )
    res = am.fit(disp="off")

    desc = (
        f"GARCH(1,1), mean={mean}, dist={dist}, "
        f"scale_factor={scale_factor}, deseasonalise={deseasonalise}"
    )

    return GarchFit(
        model_desc=desc,
        scale_factor=float(scale_factor),
        used_deseasonal=used_deseasonal,
        fit_result=res,
        returns_used=r_fit,
        intraday_scale=intraday_scale,
        deseasonal_mean=deseasonal_mean,
    )


def summarize_fit(fit: GarchFit) -> Dict[str, Any]:
    res = fit.fit_result
    params = res.params.to_dict()
    out = {
        "model": fit.model_desc,
        "nobs": int(res.nobs),
        "loglikelihood": float(res.loglikelihood),
        "aic": float(res.aic),
        "bic": float(res.bic),
        "params": params,
        "convergence_flag": int(getattr(res, "convergence_flag", -1)),
    }
    alpha = params.get("alpha[1]")
    beta = params.get("beta[1]")
    if alpha is not None and beta is not None:
        out["alpha_plus_beta"] = float(alpha + beta)
    return out


# return simulations


def _simulate_one_path_arch(res, n_steps: int, burn: int, seed: int) -> np.ndarray:
    state = np.random.get_state()
    try:
        np.random.seed(int(seed))
        sim = res.model.simulate(res.params, nobs=int(n_steps), burn=int(burn))
        return sim["data"].to_numpy(dtype=float)
    finally:
        np.random.set_state(state)


def simulate_garch_returns(
    fit: GarchFit,
    index: pd.DatetimeIndex,
    n_sims: int = 200,
    seed: int = 1,
    burn: int = 2000,
    add_back_mean: bool = True,
) -> np.ndarray:
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError("index must be a DatetimeIndex")

    n_steps = len(index)
    if n_steps <= 1:
        raise ValueError("index must have at least 2 timestamps")

    res = fit.fit_result

    if fit.used_deseasonal and fit.intraday_scale is not None:
        s_arr = scale_array(index, fit.intraday_scale)
    else:
        s_arr = np.ones(n_steps, dtype=float)

    sims = np.zeros((n_steps, n_sims), dtype=float)

    mu = float(fit.deseasonal_mean) if add_back_mean else 0.0

    for i in range(n_sims):
        r_sim_scaled = _simulate_one_path_arch(res, n_steps=n_steps, burn=burn, seed=seed + i)

        r_des = r_sim_scaled / float(fit.scale_factor)

        if add_back_mean:
            r_des = r_des + mu

        if fit.used_deseasonal:
            shock = r_des - mu
            r_out = mu + shock * s_arr
        else:
            r_out = r_des

        sims[:, i] = r_out

    return sims



# returns to prices conversions


def returns_to_price_paths(sim_returns: np.ndarray, start_price: float) -> np.ndarray:
    if start_price <= 0:
        raise ValueError("start_price must be positive")

    r = sim_returns / 100.0
    prices = float(start_price) * np.exp(np.cumsum(r, axis=0))
    return prices


# price paths diagnostics


def summarize_simulated_returns(
    sim_returns: np.ndarray,
    periods_per_year: int,
) -> pd.DataFrame:
    r = sim_returns.reshape(-1)

    summary = {
        "count": int(r.size),
        "mean": float(np.mean(r)),
        "std": float(np.std(r, ddof=0)),
        "min": float(np.min(r)),
        "p01": float(np.quantile(r, 0.01)),
        "p05": float(np.quantile(r, 0.05)),
        "median": float(np.median(r)),
        "p95": float(np.quantile(r, 0.95)),
        "p99": float(np.quantile(r, 0.99)),
        "max": float(np.max(r)),
        "skew": float(pd.Series(r).skew()),
        "kurtosis": float(pd.Series(r).kurtosis()),
    }

    return pd.DataFrame([summary])


def summarize_simulated_paths(
    sim_prices: np.ndarray,
    periods_per_year: int,
) -> pd.DataFrame:
    n_steps, n_sims = sim_prices.shape

    total_returns = np.zeros(n_sims)
    max_drawdowns = np.zeros(n_sims)
    ann_vols = np.zeros(n_sims)

    for i in range(n_sims):
        p = sim_prices[:, i]

        total_returns[i] = (p[-1] / p[0]) - 1.0

        peak = np.maximum.accumulate(p)
        dd = p / peak - 1.0
        max_drawdowns[i] = dd.min()

        r = np.diff(np.log(p))
        ann_vols[i] = np.std(r, ddof=0) * np.sqrt(periods_per_year)

    return pd.DataFrame({
        "total_return": total_returns,
        "max_drawdown": max_drawdowns,
        "ann_vol": ann_vols,
    })
