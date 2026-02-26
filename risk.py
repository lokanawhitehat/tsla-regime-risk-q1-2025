from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm


@dataclass
class VarBacktestResult:
    alpha: float
    n_obs: int
    n_violations: int
    kupiec_lr: float
    kupiec_pvalue: float
    christoffersen_lr: float
    christoffersen_pvalue: float
    mean_es: float  # average Expected Shortfall over the test window


def compute_var(
    sigma: pd.Series,
    alpha: float = 0.99,
    side: Literal["long", "short"] = "long",
) -> pd.Series:
    """
    Compute parametric VaR assuming normal returns.

    For a long position, VaR is a positive number representing loss.
    """
    z = norm.ppf(1 - alpha)  # negative for alpha>0.5
    if side == "long":
        var = -z * sigma
    else:
        var = z * sigma
    var = var.rename(f"var_{int(alpha*100)}")
    return var


def compute_es(
    sigma: pd.Series,
    alpha: float = 0.99,
    side: Literal["long", "short"] = "long",
) -> pd.Series:
    """
    Compute parametric Expected Shortfall (ES) assuming normal returns.

    ES is the expected loss given that the loss exceeds the VaR level.
    For a long position, ES is returned as a positive loss number.
    """
    z = norm.ppf(alpha)
    phi = norm.pdf(z)
    es = sigma * phi / (1 - alpha)
    if side == "short":
        es = -es
    es = es.rename(f"es_{int(alpha*100)}")
    return es


def compute_var_violations(
    returns: pd.Series,
    var: pd.Series,
) -> pd.Series:
    """
    Indicator series: 1 if realized loss exceeds VaR, else 0.
    """
    aligned = pd.concat([returns, var], axis=1).dropna()
    r = aligned.iloc[:, 0]
    v = aligned.iloc[:, 1]
    violations = (r < -v).astype(int)
    violations.name = "violation"
    return violations


def kupiec_test(
    violations: pd.Series,
    alpha: float,
) -> Tuple[float, float]:
    """
    Kupiec unconditional coverage test.
    """
    n = len(violations)
    x = int(violations.sum())
    if n == 0:
        raise ValueError("No observations provided to Kupiec test.")
    pi_hat = x / n
    if pi_hat in (0, 1):
        # avoid log(0) issues; treat as extreme
        return np.inf, 0.0

    log_l0 = x * np.log(alpha) + (n - x) * np.log(1 - alpha)
    log_l1 = x * np.log(pi_hat) + (n - x) * np.log(1 - pi_hat)
    lr = -2 * (log_l0 - log_l1)
    pvalue = 1 - chi2.cdf(lr, df=1)
    return float(lr), float(pvalue)


def christoffersen_test(
    violations: pd.Series,
) -> Tuple[float, float]:
    """
    Christoffersen independence test (1st-order Markov).
    """
    v = violations.astype(int).values
    if len(v) < 2:
        raise ValueError("Need at least 2 observations for Christoffersen test.")

    # Transition counts
    n00 = n01 = n10 = n11 = 0
    for i in range(1, len(v)):
        prev, curr = v[i - 1], v[i]
        if prev == 0 and curr == 0:
            n00 += 1
        elif prev == 0 and curr == 1:
            n01 += 1
        elif prev == 1 and curr == 0:
            n10 += 1
        else:
            n11 += 1

    n0 = n00 + n01
    n1 = n10 + n11
    if n0 == 0 or n1 == 0:
        return np.inf, 0.0

    p01 = n01 / n0
    p11 = n11 / n1
    p = (n01 + n11) / (n0 + n1)

    def safe_log(x):
        return np.log(x) if 0 < x < 1 else 0.0

    log_l0 = (
        n00 * safe_log(1 - p)
        + n01 * safe_log(p)
        + n10 * safe_log(1 - p)
        + n11 * safe_log(p)
    )
    log_l1 = (
        n00 * safe_log(1 - p01)
        + n01 * safe_log(p01)
        + n10 * safe_log(1 - p11)
        + n11 * safe_log(p11)
    )

    lr = -2 * (log_l0 - log_l1)
    pvalue = 1 - chi2.cdf(lr, df=1)
    return float(lr), float(pvalue)


def backtest_var(
    returns: pd.Series,
    sigma: pd.Series,
    alpha: float = 0.99,
) -> VarBacktestResult:
    """
    Full VaR backtest: VaR + violations + Kupiec + Christoffersen.
    """
    var = compute_var(sigma, alpha=alpha)
    es = compute_es(sigma, alpha=alpha)

    # Align ES with returns index and take the average over the test window
    es_aligned = es.reindex(returns.index)
    mean_es = float(es_aligned.mean())

    violations = compute_var_violations(returns, var)
    lr_uc, p_uc = kupiec_test(violations, alpha=alpha)
    lr_ind, p_ind = christoffersen_test(violations)

    return VarBacktestResult(
        alpha=alpha,
        n_obs=len(violations),
        n_violations=int(violations.sum()),
        kupiec_lr=lr_uc,
        kupiec_pvalue=p_uc,
        christoffersen_lr=lr_ind,
        christoffersen_pvalue=p_ind,
        mean_es=mean_es,
    )


# ---- Volatility forecast accuracy ----

def volatility_forecast_losses(
    returns: pd.Series,
    sigma_forecast: pd.Series,
) -> pd.DataFrame:
    """
    Align returns and sigma_forecast, compute realized variance proxy (r²),
    then return a DataFrame with per-observation MSE and QLIKE for variance.
    """
    aligned = pd.concat([returns, sigma_forecast], axis=1).dropna()
    r = aligned.iloc[:, 0].values
    h_pred = (aligned.iloc[:, 1].values) ** 2  # forecast variance
    h_true = r ** 2  # realized variance proxy (squared return)

    # Avoid zeros for QLIKE
    h_pred = np.maximum(h_pred, 1e-10)
    h_true = np.maximum(h_true, 1e-10)

    mse = (h_pred - h_true) ** 2
    # QLIKE: (h_true/h_pred) - log(h_true/h_pred) - 1
    qlike = (h_true / h_pred) - np.log(h_true / h_pred) - 1.0

    return pd.DataFrame(
        {"mse": mse, "qlike": qlike},
        index=aligned.index,
    )


def mean_squared_error_variance(
    returns: pd.Series,
    sigma_forecast: pd.Series,
) -> float:
    """Mean squared error of variance forecast vs squared return."""
    df = volatility_forecast_losses(returns, sigma_forecast)
    return float(df["mse"].mean())


def mean_qlike_loss(
    returns: pd.Series,
    sigma_forecast: pd.Series,
) -> float:
    """Mean QLIKE loss for variance (lower is better)."""
    df = volatility_forecast_losses(returns, sigma_forecast)
    return float(df["qlike"].mean())


def diebold_mariano_test(
    returns: pd.Series,
    sigma_forecast_a: pd.Series,
    sigma_forecast_b: pd.Series,
    loss: Literal["mse", "qlike"] = "mse",
) -> Tuple[float, float]:
    """
    Diebold-Mariano test for equal predictive ability.
    H0: E[L(e_A) - L(e_B)] = 0. Returns (DM statistic, one-sided p-value).
    Positive DM and p < 0.05 means model B is significantly better (lower loss).
    """
    df_a = volatility_forecast_losses(returns, sigma_forecast_a)
    df_b = volatility_forecast_losses(returns, sigma_forecast_b)
    # Align
    common = df_a.join(df_b, lsuffix="_a", rsuffix="_b").dropna()
    if len(common) < 2:
        return np.nan, np.nan
    loss_a = common["mse_a"] if loss == "mse" else common["qlike_a"]
    loss_b = common["mse_b"] if loss == "mse" else common["qlike_b"]
    d = (loss_a - loss_b).values
    n = len(d)
    d_bar = d.mean()
    # HAC variance with truncation at n^(1/4) or similar
    k = max(1, int(n ** 0.25))
    gamma_0 = np.var(d, ddof=1)
    for lag in range(1, k):
        gamma_0 += 2 * (1 - lag / (k + 1)) * np.cov(d[:-lag], d[lag:])[0, 1]
    var_d = gamma_0 / n
    if var_d <= 0:
        return np.nan, np.nan
    dm = d_bar / np.sqrt(var_d)
    pvalue = 2 * (1 - norm.cdf(abs(dm)))
    return float(dm), float(pvalue)

