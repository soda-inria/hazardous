"""
Debiased Scoring loss
=====================

With the synthetic dataset, we can obtain the true distribution of the
censoring events, along with the shape and the scale of the weibull distribution
for each sample.

With the true distribution of censored events, we can compute the debiased loss
to evaluate and compare models.
"""

import numpy as np
import pandas as pd
from scipy.stats import weibull_min
from sklearn.utils import Bunch

from hazardous.utils import check_y_survival


def compute_true_probas(
    y_censored,
    shape_censoring,
    scale_censoring,
    n_time_grid_steps=100,
):
    """

    Parameters
    ----------
    TODO

    Returns
    -------
    bunch : scikit-learn Bunch
        Bunch object with the following items:

        * time_grid : ndarray of shape (n_time_grid_steps, )
          The time grid used to generate censored_proba_time_grid
    """
    shape_censoring = np.atleast_1d(shape_censoring)
    scale_censoring = np.atleast_1d(scale_censoring)

    event = y_censored["event"]
    duration = y_censored["duration"]

    any_event_mask = event > 0
    observed_times = duration[any_event_mask]
    quantile_grid = np.linspace(0, 1, num=n_time_grid_steps)
    time_grid = np.quantile(observed_times, q=quantile_grid)

    censored_proba_time_grid = _generate_survival_proba(
        time_grid, shape_censoring, scale_censoring
    )
    censored_proba_duration = []
    censored_proba_duration = 1 - pd.DataFrame(
        weibull_min.cdf(y_censored["duration"], shape_censoring, scale=scale_censoring)
    )  # (G^*(t_i |x_i)) for all x_i

    return Bunch(
        censored_proba_time_grid=censored_proba_time_grid,
        censored_proba_duration=censored_proba_duration,
        time_grid=time_grid,
    )


def _generate_survival_proba(time_steps, shape, scale):
    """Return 1 - CDF of a Weibull distribution for some given time steps."""
    survival_proba = []
    for time_step in time_steps:
        incidence_proba = weibull_min.cdf(time_step, shape, scale=scale)
        survival_proba.append(1 - incidence_proba)
    return pd.DataFrame(survival_proba).T  # shape: (n_samples, n_time_steps)


def brier_score_true_probas(
    y_true,
    y_pred,
    time_grid,
    event_of_interest,
    censored_proba_duration,
    censored_proba_time_grid,
):
    event_true, duration_true = check_y_survival(y_true)
    if event_of_interest == "any":
        event_true = event_true > 0
        event_of_interest = 1

    n_samples = event_true.shape[0]
    n_time_steps = time_grid.shape[0]
    brier_scores = np.empty(
        shape=(n_samples, n_time_steps),
        dtype=np.float64,
    )
    for t_idx, time_step in enumerate(time_grid):
        y_true_binary, weights = compute_weights_and_target(
            event_true,
            duration_true,
            time_step,
            event_of_interest,
            censored_proba_duration,
            censored_proba_time_grid[:, t_idx],
        )
        squared_error = (y_true_binary - y_pred[:, t_idx]) ** 2
        brier_scores[:, t_idx] = weights * squared_error

    return brier_scores.mean(axis=0)


def compute_weights_and_target(
    event_true,
    duration_true,
    time_step,
    event_of_interest,
    censored_proba_duration,  # (n_samples x 1 ) (G^*(t_i| x_i))
    censored_proba_at_time,  # (n_samples x 1) (G^*(\tau | x_i))
):
    """Compute the binary event indicator and IPCW at time_step.

    Parameters
    ----------
    TODO

    Returns
    -------
    TODO
    """
    y_true_binary = (
        (event_true == event_of_interest) & (duration_true <= time_step)
    ).astype(np.int32)

    at_risk = duration_true > time_step
    ipcw_time_grid = (1 / censored_proba_at_time,)
    weights = np.where(at_risk, ipcw_time_grid, 0)

    ipcw_y_duration = 1 / censored_proba_duration
    any_event_observed = (event_true > 0) & (duration_true <= time_step)
    weights = np.where(any_event_observed, ipcw_y_duration, weights)

    return y_true_binary, weights


def integrated_scoring_metric(scores, times):
    ordering = np.argsort(times)
    sorted_times = times[ordering]
    sorted_scores = scores[ordering]
    time_span = sorted_times[-1] - sorted_times[0]
    return np.trapz(sorted_scores, sorted_times) / time_span
