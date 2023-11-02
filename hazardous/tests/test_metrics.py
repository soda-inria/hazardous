import re
from functools import lru_cache

import numpy as np
import pandas as pd
import pytest
from lifelines import CoxPHFitter
from lifelines.datasets import load_regression_dataset
from numpy.testing import assert_allclose
from scipy.interpolate import interp1d
from scipy.stats import weibull_min

from ..metrics import (
    brier_score_incidence,
    brier_score_survival,
    integrated_brier_score_incidence,
    integrated_brier_score_survival,
)
from ..metrics._brier_score import IncidenceScoreComputer
from ..utils import _dict_to_pd, _dict_to_recarray

X = load_regression_dataset()
X_train, X_test = X.iloc[:150], X.iloc[150:]
y_train = dict(
    event=X_train["E"].to_numpy(),
    duration=X_train["T"].to_numpy(),
)
y_test = dict(
    event=X_test["E"].to_numpy(),
    duration=X_test["T"].to_numpy(),
)
times = np.arange(
    y_test["duration"].min(),
    y_test["duration"].max() - 1,
)

est = CoxPHFitter().fit(X_train, duration_col="T", event_col="E")
y_pred_survival = est.predict_survival_function(X_test, times)
y_pred_survival = y_pred_survival.T.values  # (n_samples, n_times)

# Expected BS survival values computed with scikit-survival:
#
# from sksurv.metrics import brier_score as brier_score_sksurv
# from sksurv.metrics import integrated_brier_score as integrated_brier_score_sksurv
# from hazardous.utils import _dict_to_recarray
# from pprint import pprint
#
# _, bs_from_sksurv = brier_score_sksurv(
#     _dict_to_recarray(y_train, cast_event_to_bool=True),
#     _dict_to_recarray(y_test, cast_event_to_bool=True),
#     y_pred_survival,
#     times,
# )
# pprint(bs_from_sksurv.tolist())
# ibs_from_sksurv = integrated_brier_score_sksurv(
#     _dict_to_recarray(y_train, cast_event_to_bool=True),
#     _dict_to_recarray(y_test, cast_event_to_bool=True),
#     y_pred_survival,
#     times,
# )
# print(ibs_from_sksurv)

EXPECTED_BS_SURVIVAL_FROM_SKSURV = np.array(
    [
        0.019210159012786377,
        0.08987547845995612,
        0.11693114655908207,
        0.1883220229893822,
        0.2134659930805141,
        0.24300206373683012,
        0.242177758373255,
        0.2198792376648805,
        0.199871735321175,
        0.16301317649264274,
        0.07628880587676132,
        0.05829175905913857,
        0.0663998034737539,
        0.04524901436623458,
        0.045536886754156194,
        0.022500377138006216,
        0.022591326598969338,
    ]
)
EXPECTED_IBS_SURVIVAL_FROM_SKSURV = 0.12573162513447791


def test_brier_score_survival_sksurv_consistency():
    loss = brier_score_survival(
        y_train,
        y_test,
        y_pred_survival,
        times,
    )
    assert_allclose(loss, EXPECTED_BS_SURVIVAL_FROM_SKSURV, atol=1e-6)


def test_integrated_brier_score_survival_sksurv_consistency():
    ibs = integrated_brier_score_survival(
        y_train,
        y_test,
        y_pred_survival,
        times,
    )
    assert ibs == pytest.approx(EXPECTED_IBS_SURVIVAL_FROM_SKSURV, abs=1e-6)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_integrated_brier_score_on_shuffled_times(seed):
    # Check that IBS computation is invariant to the order of the times
    ibs_ref = integrated_brier_score_survival(
        y_train,
        y_test,
        y_pred_survival,
        times,
    )

    rng = np.random.default_rng(seed)
    perm_indices = rng.permutation(times.shape[0])
    times_shuffled = times[perm_indices]
    y_pred_survival_shuffled = y_pred_survival[:, perm_indices]

    ibs_shuffled = integrated_brier_score_survival(
        y_train,
        y_test,
        y_pred_survival_shuffled,
        times_shuffled,
    )
    assert ibs_shuffled == pytest.approx(ibs_ref)


@pytest.mark.parametrize("event_of_interest", [1, "any"])
def test_brier_score_incidence_survival_equivalence(event_of_interest):
    loss_survival = brier_score_survival(
        y_train,
        y_test,
        y_pred_survival,
        times,
    )
    loss_incidence = brier_score_incidence(
        y_train,
        y_test,
        1 - y_pred_survival,
        times,
        event_of_interest,
    )
    assert_allclose(loss_survival, loss_incidence)

    ibs_survival = integrated_brier_score_survival(
        y_train,
        y_test,
        y_pred_survival,
        times,
    )
    ibs_incidence = integrated_brier_score_incidence(
        y_train,
        y_test,
        1 - y_pred_survival,
        times,
        event_of_interest,
    )
    assert ibs_survival == pytest.approx(ibs_incidence)


def test_brier_score_warnings_on_competive_event():
    coef = np.random.choice([1, 2], size=y_train["event"].shape[0])
    y_train["event"] *= coef

    msg = "Computing the survival Brier score only makes sense"
    with pytest.warns(match=msg):
        IncidenceScoreComputer(
            y_train,
            event_of_interest=2,
        ).brier_score_survival(
            y_test,
            y_pred_survival,
            times,
        )


@pytest.mark.parametrize("event_of_interest", [-10, 0])
def test_brier_score_incidence_wrong_parameters_value_error(event_of_interest):
    msg = "event_of_interest must be a strictly positive integer or 'any'"
    with pytest.raises(ValueError, match=msg):
        brier_score_incidence(
            y_train,
            y_test,
            y_pred_survival,
            times,
            event_of_interest,
        )


@pytest.mark.parametrize("event_of_interest", [-10, 0, "wrong_event"])
def test_brier_score_incidence_wrong_parameters_type_error(event_of_interest):
    msg = "event_of_interest must be an instance of"
    for event_of_interest in [None, [1], (2, 3)]:
        with pytest.raises(TypeError, match=msg):
            brier_score_incidence(
                y_train,
                y_test,
                y_pred_survival,
                times,
                event_of_interest,
            )


@pytest.mark.parametrize("format_func", [_dict_to_pd, _dict_to_recarray])
def test_test_brier_score_survival_inputs_format(format_func):
    loss = brier_score_survival(
        format_func(y_train),
        format_func(y_test),
        y_pred_survival,
        times,
    )
    assert_allclose(loss, EXPECTED_BS_SURVIVAL_FROM_SKSURV, atol=1e-6)


def test_brier_score_survival_wrong_inputs():
    y_train_wrong = dict(
        wrong_name=y_train["event"],
        duration=y_train["duration"],
    )
    msg = (
        "y must be a record array, a pandas DataFrame, or a dict whose dtypes, "
        "keys or columns are 'event' and 'duration'."
    )
    with pytest.raises(ValueError, match=msg):
        brier_score_survival(
            y_train_wrong,
            y_test,
            y_pred_survival,
            times,
        )

    msg = "'times' length (5) must be equal to y_pred.shape[1] (17)."
    with pytest.raises(ValueError, match=re.escape(msg)):
        brier_score_survival(
            y_train,
            y_test,
            y_pred_survival,
            times[:5],
        )


def _weibull_hazard(t, shape=1.0, scale=1.0):
    """Weibull hazard function.

    See: https://en.wikipedia.org/wiki/Weibull_distribution

    This implementation plugs an arbitrary null hazard value at `t==0` because
    fractional powers of 0 are undefined.

    This does not seem to be mathematically correct but in practice it does
    make it possible to integrate the hazard function into cumulative incidence
    functions in a way that matches the Aalen-Johansen estimator.

    See examples/plot_marginal_cumulative_incidence_estimation.py for a
    visual confirmation.
    """
    with np.errstate(divide="ignore"):
        weibull_hazard_at_t = (shape / scale) * (t / scale) ** (shape - 1.0)
    return np.where(t == 0, 0.0, weibull_hazard_at_t)


def _sample_marginal_competing_weibull_data(
    distribution_params,
    n_samples,
    random_state=None,
):
    event_times = np.concatenate(
        [
            weibull_min.rvs(
                dist["params"]["shape"],
                scale=dist["params"]["scale"],
                size=n_samples,
                random_state=random_state,
            ).reshape(-1, 1)
            for dist in distribution_params
        ],
        axis=1,
    )
    event_ids = np.asarray([d["event_id"] for d in distribution_params])
    first_event_idx = np.argmin(event_times, axis=1)

    y = pd.DataFrame(
        dict(
            event=event_ids[first_event_idx],
            duration=event_times[np.arange(n_samples), first_event_idx],
        )
    )
    return y


def _integrate_weibull_incidence_curves(distributions, t_max):
    """Numerically integrate the Weibull hazard functions to get the CIFs.

    The numerical integration is performed using a fine time grid and the

    Note: in a competing risks setting, the cause-specific hazard functions are
    coupled: they do not only depend on the distribution parameters that
    determine the cause-specific hazards but also on the aggregate survival to
    all competing events. This is why we need to integrate the hazard functions
    to get the CIFs instead of just considering the parametric defintion of the
    cumulative density functions (CDFs) of the Weibull distribution.
    """
    fine_time_grid = np.linspace(0, t_max, num=10_000_000)
    dt = np.diff(fine_time_grid)[0]
    all_hazards = np.stack(
        [
            _weibull_hazard(fine_time_grid, **dist["params"])
            for dist in distributions
            if dist["event_id"] > 0  # skip censoring distribution
        ],
        axis=0,
    )
    any_event_hazards = all_hazards.sum(axis=0)
    any_event_survival = np.exp(-(any_event_hazards.cumsum(axis=-1) * dt))
    cifs = [
        ((hazards_i * any_event_survival).cumsum(axis=-1) * dt)
        for hazards_i in all_hazards
    ]
    downscale_factor = fine_time_grid.size // 1000
    return (
        interp1d(
            fine_time_grid[::downscale_factor],
            any_event_survival[::downscale_factor],
            kind="previous",
            bounds_error=False,
            fill_value="extrapolate",
        ),
        [
            interp1d(
                fine_time_grid[::downscale_factor],
                cif[::downscale_factor],
                kind="previous",
                bounds_error=False,
                fill_value="extrapolate",
            )
            for cif in cifs
        ],
    )


@lru_cache()
def _sample_test_data(
    n_samples=10_000, base_scale=1000.0, censored=True, random_state=None
):
    distribution_params = [
        {"event_id": 0, "params": {"scale": 3 * base_scale, "shape": 1}},  # censoring
        {"event_id": 1, "params": {"scale": 10 * base_scale, "shape": 0.5}},
        {"event_id": 2, "params": {"scale": 3 * base_scale, "shape": 1}},
        {"event_id": 3, "params": {"scale": 3 * base_scale, "shape": 5}},
    ]

    if not censored:
        distribution_params = distribution_params[1:]

    data = _sample_marginal_competing_weibull_data(
        distribution_params, n_samples=n_samples, random_state=random_state
    )
    t_max = data.query("event > 0")["duration"].max()
    true_survival, true_cifs = _integrate_weibull_incidence_curves(
        distribution_params, t_max
    )

    return data, true_survival, true_cifs, t_max


@pytest.mark.parametrize("censored", [True, False])
@pytest.mark.parametrize("relative_time_horizon", [0.2, 0.5, 0.8])
@pytest.mark.parametrize("seed", range(2))
def test_brier_score_optimality_survival(censored, relative_time_horizon, seed):
    n_samples = 100_000
    y, true_survival, _, t_max = _sample_test_data(
        n_samples=n_samples, censored=censored, random_state=seed
    )
    if censored:
        assert y["event"].min() == 0
    else:
        assert y["event"].min() > 0

    # Convert competing event analysis data to data suitable for "any event"
    # survival analysis by ignoring the event_id types.
    y["event"] = (y["event"] > 0).astype(np.int32)

    time_horizon = t_max * relative_time_horizon
    expected_optimal_estimate = true_survival(time_horizon)
    expected_best_bs = brier_score_survival(
        y,
        y,
        np.full(shape=(n_samples, 1), fill_value=expected_optimal_estimate),
        times=np.asarray([time_horizon]),
    )[0]
    survival_prob_grid = np.linspace(0, 1, num=11)
    bs_values = np.asarray(
        [
            brier_score_survival(
                y,
                y,
                np.full(shape=(n_samples, 1), fill_value=y_pred),
                times=np.asarray([time_horizon]),
            )
            for y_pred in survival_prob_grid
        ]
    ).ravel()
    assert bs_values.min() >= expected_best_bs

    # Check that the Brier score makes it possible to discriminate between
    # different survival probabilities estimate.
    best_prob_estimate = survival_prob_grid[bs_values.argmin()]
    assert best_prob_estimate == pytest.approx(expected_optimal_estimate, abs=0.1)


@pytest.mark.parametrize("event_of_interest", [1, 2, 3])
@pytest.mark.parametrize("censored", [True, False])
@pytest.mark.parametrize("relative_time_horizon", [0.1, 0.5, 0.9])
@pytest.mark.parametrize("seed", range(2))
def test_brier_score_optimality_competing_risks(
    event_of_interest, censored, relative_time_horizon, seed
):
    # TODO: adapt test_brier_score_optimality_survival to the competing risks
    # setting.
    pass
