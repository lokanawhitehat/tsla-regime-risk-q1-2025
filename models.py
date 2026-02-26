from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from arch import arch_model
from hmmlearn.hmm import GaussianHMM


@dataclass
class GarchResult:
    model: any
    volatility: pd.Series


@dataclass
class RegimeResult:
    model: GaussianHMM
    states: pd.Series
    state_probs: pd.DataFrame


@dataclass
class MsGarchResult:
    """
    Simple Markov-switching-style GARCH result.

    This is a pragmatic approximation:
    - fit a baseline GARCH to the full return series
    - fit a Markov-switching model (HMM) to returns
    - rescale the GARCH volatility by regime-specific factors
    """

    garch: GarchResult
    regime: RegimeResult
    ms_volatility: pd.Series


def fit_garch(
    returns: pd.Series,
    p: int = 1,
    q: int = 1,
    dist: str = "normal",
) -> GarchResult:
    """
    Fit a simple GARCH(p, q) model to returns.
    """
    r = returns.dropna() * 100  # convert to percent to stabilize arch
    am = arch_model(r, p=p, q=q, vol="GARCH", dist=dist, mean="Constant")
    res = am.fit(disp="off")
    cond_vol = res.conditional_volatility / 100.0  # back to return units
    cond_vol.index = returns.dropna().index
    return GarchResult(model=res, volatility=cond_vol)


def forecast_garch_vol(
    garch_result: GarchResult,
    horizon: int = 1,
) -> float:
    """
    One-step-ahead (or short horizon) volatility forecast.
    """
    fcast = garch_result.model.forecast(horizon=horizon)
    sigma = np.sqrt(fcast.variance.values[-1, -1]) / 100.0
    return float(sigma)


def fit_markov_regime_switching(
    returns: pd.Series,
    n_states: int = 2,
    covariance_type: str = "diag",
    random_state: Optional[int] = 42,
) -> RegimeResult:
    """
    Approximate a regime-switching volatility model using a Gaussian HMM
    fitted to daily returns.

    This is not a full Markov-Switching GARCH, but it captures:
    - multiple return/volatility regimes
    - a transition matrix between regimes
    """
    r = returns.dropna().values.reshape(-1, 1)
    hmm = GaussianHMM(
        n_components=n_states,
        covariance_type=covariance_type,
        n_iter=500,
        random_state=random_state,
    )
    hmm.fit(r)

    hidden_states = hmm.predict(r)
    state_probs = hmm.predict_proba(r)

    idx = returns.dropna().index
    states_series = pd.Series(hidden_states, index=idx, name="state")

    probs_df = pd.DataFrame(state_probs, index=idx)
    probs_df.columns = [f"state_{i}_prob" for i in range(n_states)]

    return RegimeResult(model=hmm, states=states_series, state_probs=probs_df)


def fit_ms_garch_like(
    returns: pd.Series,
    n_states: int = 2,
    covariance_type: str = "diag",
    random_state: Optional[int] = 42,
) -> MsGarchResult:
    """
    Construct a regime-switching-style GARCH volatility series.

    Steps:
    1. Fit baseline GARCH on the full return history.
    2. Fit a Gaussian HMM (regime model) on the same returns.
    3. For each regime, compute the standard deviation of returns in that regime.
    4. Scale the GARCH volatility by (regime_std / overall_std).
    """
    base_garch = fit_garch(returns)
    regime = fit_markov_regime_switching(
        returns,
        n_states=n_states,
        covariance_type=covariance_type,
        random_state=random_state,
    )

    overall_std = returns.dropna().std()
    if overall_std == 0 or np.isnan(overall_std):
        overall_std = 1.0

    factors = {}
    for state_id in range(n_states):
        mask = regime.states == state_id
        if mask.any():
            state_std = returns[mask].dropna().std()
            if state_std and not np.isnan(state_std):
                factors[state_id] = float(state_std / overall_std)
            else:
                factors[state_id] = 1.0
        else:
            factors[state_id] = 1.0

    # Align GARCH volatility index with regime states
    sigma = base_garch.volatility.reindex(regime.states.index)
    factor_series = regime.states.map(lambda s: factors.get(int(s), 1.0))
    ms_sigma = sigma * factor_series
    ms_sigma.name = "ms_garch_sigma"

    return MsGarchResult(garch=base_garch, regime=regime, ms_volatility=ms_sigma)


def current_regime_info(regime_result: RegimeResult) -> Tuple[int, float]:
    """
    Convenience helper: get the latest most likely regime and its probability.
    """
    if regime_result.states.empty:
        raise ValueError("Regime states are empty.")
    last_state = int(regime_result.states.iloc[-1])
    prob_col = f"state_{last_state}_prob"
    last_prob = float(regime_result.state_probs[prob_col].iloc[-1])
    return last_state, last_prob

