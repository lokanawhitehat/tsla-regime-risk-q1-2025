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

