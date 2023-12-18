## Debiased Scoring loss
# With the synthetic dataset, one can obtain the true distribution of the
# censoring events.
# With the synthetic dataset, we can obtain the shape and the scale of the weibull
# distribution for each sample.
# With the true distribution of censored events, we can compute the debiased loss
# to evaluate and compare models.

import numpy as np
import pandas as pd
from scipy.stats import weibull_min
from sklearn.datasets._base import Bunch

from hazardous.utils import check_y_survival

seed = 0
n_time_grid_steps = 100
independent_censoring = True


# %%
def compute_true_probas(
    y_censored,
    shape_censoring,
    scale_censoring,
    independent_censoring=independent_censoring,
):
    if independent_censoring:
        params = pd.DataFrame(
            np.array([[shape_censoring], [scale_censoring]]).T,
            columns=["shape", "scale"],
        )
    else:
        params = pd.DataFrame(
            np.array([shape_censoring, scale_censoring]).T,
            columns=["shape", "scale"],
        )

    event = y_censored["event"]
    duration = y_censored["duration"]

    any_event_mask = event > 0
    observed_times = duration[any_event_mask]
    time_grid = np.quantile(observed_times, np.linspace(0, 1, num=n_time_grid_steps))

    censored_proba_time_grid = []
    for time in time_grid:
        proba = weibull_min.cdf(time, params["shape"], scale=params["scale"])
        censored_proba_time_grid.append(pd.DataFrame(proba, columns=[time]))

    censored_proba_time_grid = 1 - pd.concat(censored_proba_time_grid, axis=1)

    censored_proba_duration = []
    censored_proba_duration = 1 - pd.DataFrame(
        weibull_min.cdf(y_censored["duration"], params["shape"], scale=params["scale"])
    )
    return Bunch(
        censored_proba_time_grid=censored_proba_time_grid,
        censored_proba_duration=censored_proba_duration,
        time_grid=time_grid,
    )


# computing loss
def compute_weights_and_target(
    event_true,
    duration_true,
    time,
    t_idx,
    k,
    censored_proba_duration,
    censored_proba_time_grid,
):
    event_k_before_horizon = (event_true == k) & (duration_true <= time)
    y_true_binary = event_k_before_horizon.astype(np.int32)

    any_event_or_censoring_after_horizon = duration_true > time
    ipcw_times = 1 / censored_proba_time_grid.iloc[:, t_idx]
    weights = np.where(any_event_or_censoring_after_horizon, ipcw_times, 0)
    ipcw_y_duration = 1 / censored_proba_duration.values.flatten()
    any_observed_event_before_horizon = (event_true > 0) & (duration_true <= time)
    weights = np.where(any_observed_event_before_horizon, ipcw_y_duration, weights)
    return y_true_binary, weights


def brier_score_true_probas(
    y_true,
    y_pred,
    times,
    event_of_interest,
    censored_proba_duration,
    censored_proba_time_grid,
):
    event_true, duration_true = check_y_survival(y_true)
    if event_of_interest == "any":
        event_true = event_true > 0

    n_samples = event_true.shape[0]
    n_time_steps = times.shape[0]
    brier_scores = np.empty(
        shape=(n_samples, n_time_steps),
        dtype=np.float64,
    )

    for t_idx, t in enumerate(times):
        if event_of_interest == "any":
            # y should already be provided as binary indicator
            k = 1
        else:
            k = event_of_interest

        time = np.full(shape=n_samples, fill_value=t)
        y_true_binary, weights = compute_weights_and_target(
            event_true,
            duration_true,
            time,
            t_idx,
            k,
            censored_proba_duration,
            censored_proba_time_grid,
        )
        squared_error = (y_true_binary - y_pred[:, t_idx]) ** 2
        brier_scores[:, t_idx] = weights * squared_error

    return brier_scores.mean(axis=0)


def integrated_scoring_metric(scores, times):
    ordering = np.argsort(times)
    sorted_times = times[ordering]
    sorted_scores = scores[ordering]
    time_span = sorted_times[-1] - sorted_times[0]
    return np.trapz(sorted_scores, sorted_times) / time_span
