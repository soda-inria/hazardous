## Debiased Scoring loss
# With the synthetic dataset, one can obtain the true distribution of the
# censoring events.
# With the synthetic dataset, we can obtain the shape and the scale of the weibull
# distribution for each sample.
# With the true distribution of censored events, we can compute the debiased loss
# to evaluate and compare models.

import matplotlib.pyplot as plt

# %%
import numpy as np
import pandas as pd
from scipy.stats import weibull_min
from sklearn.datasets._base import Bunch

import hazardous.data._competing_weibull as competing_w
from hazardous import GradientBoostingIncidence
from hazardous.metrics._brier_score import brier_score_incidence
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


# %%
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
    # import ipdb; ipdb.set_trace()
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


# %%

if __name__ == "__main__":
    # Here, we create a synthetic dataset, either with a censoring independant
    # of the covariates of dependant with a rate given by the user.
    # We train a GBI to obtain some prediction. Givent a time grid, we compute
    # the true Brier score (with the real distribution of the censoring) and the BS
    # with an estimate of the proba of censoring (Kapplan Meier).
    #

    independent_censoring = True
    event_of_interest = 1

    X, _, y_uncensored = competing_w.make_synthetic_competing_weibull(
        n_samples=1_000_000,
        base_scale=1_000,
        n_features=10,
        features_rate=0.3,
        degree_interaction=2,
        independent_censoring=independent_censoring,
        features_censoring_rate=0.2,
        return_uncensored_data=True,
        return_X_y=True,
        feature_rounding=None,
        target_rounding=None,
        censoring_relative_scale=None,
        random_state=seed,
        complex_features=True,
    )

    bunch_data = competing_w._censor(
        y_uncensored,
        X=X,
        return_params_censo=True,
        independent_censoring=independent_censoring,
        features_censoring_rate=0.5,
        censoring_relative_scale=0.5,
        random_state=seed,
    )
    y_censored = bunch_data.y_censored
    shape_censoring = bunch_data.shape_censoring
    scale_censoring = bunch_data.scale_censoring
    probas = compute_true_probas(
        y_censored,
        shape_censoring,
        scale_censoring,
        independent_censoring=independent_censoring,
    )
    time_grid = probas.time_grid
    censored_proba_duration = probas.censored_proba_duration
    censored_proba_time_grid = probas.censored_proba_time_grid

    gbi = GradientBoostingIncidence(
        learning_rate=0.1,
        n_iter=20,
        max_leaf_nodes=15,
        hard_zero_fraction=0.1,
        min_samples_leaf=5,
        loss="ibs",
        show_progressbar=False,
        random_state=seed,
        event_of_interest=event_of_interest,
    )
    gbi.fit(X, y_censored)

    y_pred = gbi.predict_cumulative_incidence(X, times=time_grid)

    scores = brier_score_true_probas(
        y_censored,
        y_pred,
        times=time_grid,
        event_of_interest=event_of_interest,
        censored_proba_duration=censored_proba_duration,
        censored_proba_time_grid=censored_proba_time_grid,
    )

    bs_scores = brier_score_incidence(
        y_censored, y_censored, y_pred, time_grid, event_of_interest=event_of_interest
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(time_grid, scores, label="from true distrib")
    ax.plot(time_grid, bs_scores, label="from estimate distrib with km")

    ax.set(
        title="Time-varying Brier score",
    )
    ax.legend()
    ax.show()
