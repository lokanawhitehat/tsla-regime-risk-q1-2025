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
    )

