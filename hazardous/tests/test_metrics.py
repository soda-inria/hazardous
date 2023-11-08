import re
from functools import lru_cache

import numpy as np
import pandas as pd
import pytest
from lifelines import AalenJohansenFitter, CoxPHFitter, KaplanMeierFitter
from lifelines.datasets import load_regression_dataset
from numpy.testing import assert_allclose
from scipy.interpolate import interp1d
from scipy.stats import weibull_min
from sklearn.utils import check_random_state

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
    make it more convenient to integrate the hazard function into cumulative
    incidence functions in a way that matches the Aalen-Johansen estimator.

    See examples/plot_marginal_cumulative_incidence_estimation.py for a visual
    confirmation.
    """
    with np.errstate(divide="ignore"):
        weibull_hazard_at_t = (shape / scale) * (t / scale) ** (shape - 1.0)
    return np.where(t == 0, 0.0, weibull_hazard_at_t)


def _sample_marginal_competing_weibull_data(
    distributions,
    n_samples,
    random_state=None,
):
    # It's important to use the same RNG instance to sample different events to
    # avoid sampling the same event times multiple times in case of repeated
    # distribution parameters. Otherwise we would introduce a bias when
    # computing the argmin.
    rng = check_random_state(random_state)
    event_times = np.stack(
        [
            weibull_min.rvs(
                dist["params"]["shape"],
                scale=dist["params"]["scale"],
                size=n_samples,
                random_state=rng,
            )
            for dist in distributions
        ],
        axis=0,
    )
    assert event_times.shape == (len(distributions), n_samples)
    event_ids = np.asarray([d["event_id"] for d in distributions])
    first_event_indices = np.argmin(event_times, axis=0)
    assert first_event_indices.shape == (n_samples,)

    y = pd.DataFrame(
        dict(
            event=event_ids[first_event_indices],
            duration=event_times[first_event_indices, np.arange(n_samples)],
        )
    )
    return y


def _integrate_weibull_incidence_curves(distribution_params, t_max):
    """Numerically integrate the Weibull hazard functions to get the CIFs.

    The numerical integration is performed using a very fine time grid and
    then down-sampled to a coarser time grid to build the returned time
    interpolators.

    Note: in a competing risks setting, the cause-specific incidence functions are
    coupled: they do not only depend on the distribution parameters that
    determine the cause-specific hazards but also on the aggregate survival to
    all competing events. This is why we need to integrate the hazard functions
    to get the CIFs instead of just considering the parametric definition of the
    cumulative density functions (CDFs) of the Weibull distribution.
    """
    fine_time_grid = np.linspace(0, 1.1 * t_max, num=1_000_000)
    dt = np.diff(fine_time_grid)[0]
    all_hazards = np.stack(
        [_weibull_hazard(fine_time_grid, **params) for params in distribution_params],
        axis=0,
    )
    assert all_hazards.shape == (len(distribution_params), fine_time_grid.size)
    any_event_hazards = all_hazards.sum(axis=0)
    any_event_survival = np.exp(-(any_event_hazards[1:].cumsum(axis=-1) * dt))
    any_event_survival = np.concatenate([[1.0], any_event_survival])
    cifs = np.stack(
        [
            ((hazards_i[1:] * any_event_survival[:-1]).cumsum(axis=-1) * dt)
            for hazards_i in all_hazards
        ],
        axis=0,
    )
    cifs = np.concatenate(
        [
            np.zeros(shape=(cifs.shape[0], 1)),
            cifs,
        ],
        axis=1,
    )
    # Downscale the survival and CIFs to make the interpolation faster and more
    # memory efficient.
    downscale_factor = fine_time_grid.size // 10_000
    return (
        interp1d(
            fine_time_grid[::downscale_factor].copy(),
            any_event_survival[::downscale_factor].copy(),
            kind="previous",
            bounds_error=False,
            fill_value="extrapolate",
        ),
        [
            interp1d(
                fine_time_grid[::downscale_factor].copy(),
                cif[::downscale_factor].copy(),
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
    distributions = [
        {"event_id": 1, "params": {"scale": 10 * base_scale, "shape": 0.5}},
        {"event_id": 2, "params": {"scale": 3 * base_scale, "shape": 1}},
        {"event_id": 3, "params": {"scale": 3 * base_scale, "shape": 5}},
    ]
    if censored:
        distributions.append(
            {"event_id": 0, "params": {"scale": 2 * base_scale, "shape": 1}},
        )

    data = _sample_marginal_competing_weibull_data(
        distributions, n_samples=n_samples, random_state=random_state
    )

    # Check that we do not sample degenerate data.
    event_counts = data["event"].value_counts()
    if censored:
        assert sorted(event_counts.index) == [0, 1, 2, 3]
    else:
        assert sorted(event_counts.index) == [1, 2, 3]
    assert event_counts.min() >= 100
    assert event_counts.max() >= event_counts.min() * 1.5  # imbalanced events

    t_max = data.query("event > 0")["duration"].max()
    event_ids = [d["event_id"] for d in distributions if d["event_id"] > 0]
    true_survival, true_cifs = _integrate_weibull_incidence_curves(
        [d["params"] for d in distributions if d["event_id"] > 0], t_max
    )
    return data, true_survival, dict(zip(event_ids, true_cifs)), t_max


@pytest.mark.parametrize("censored", [True, False])
@pytest.mark.parametrize("seed", range(2))
def test_brier_score_optimality_survival(censored, seed):
    # Check that IPCW brier_score_survival behaves as a censor-agnostic proper
    # scoring rule for surival probability estimation.
    n_samples = 2_000
    y, true_survival_func, _, t_max = _sample_test_data(
        n_samples=n_samples, censored=censored, random_state=seed
    )

    # Convert competing event analysis data to data suitable for "any event"
    # survival analysis by ignoring the event_id types.
    y["event"] = (y["event"] > 0).astype(np.int32)

    # Compute the theoretical survival probability at given time horizons.
    relative_time_horizons = np.linspace(0, 1, num=11)
    time_horizons = relative_time_horizons * t_max
    true_survival_at_horizons = true_survival_func(time_horizons)

    # Check that our theoretical survival probability is compatible with the
    # Kaplan-Meier estimate. This only works for a large enough value for
    # n_samples.
    km = KaplanMeierFitter()
    km.fit(
        durations=y["duration"],
        event_observed=y["event"],
    )
    km_sf_df = km.survival_function_
    km_times = km_sf_df.index
    km_survival_probs = km_sf_df.values[:, 0]
    km_survival_func = interp1d(
        km_times,
        km_survival_probs,
        kind="previous",
        bounds_error=False,
        fill_value="extrapolate",
    )
    km_estimates = km_survival_func(time_horizons)

    # Decreasing atol would require increasing n_samples but would make the
    # rest of the test slower.
    assert_allclose(km_estimates, true_survival_at_horizons, atol=0.03)

    # Compare the Brier score computed on the sampled events for the true
    # survival probability at the given time horizons with the Brier scores
    # computed on a grid of candidate survival probability estimates.
    expected_best_bs_values = brier_score_survival(
        y,
        y,
        np.stack([true_survival_at_horizons] * n_samples, axis=0),
        times=time_horizons,
    )

    survival_prob_grid = np.linspace(0, 1, num=21)
    grid_bs_values = np.stack(
        [
            brier_score_survival(
                y,
                y,
                np.full(shape=(n_samples, time_horizons.size), fill_value=y_pred),
                times=time_horizons,
            )
            for y_pred in survival_prob_grid
        ],
        axis=0,
    )
    assert grid_bs_values.shape == (
        survival_prob_grid.size,
        relative_time_horizons.size,
    )
    # The true survival probability should be the best one:
    best_bs_values = grid_bs_values.min(axis=0)
    assert best_bs_values.shape == time_horizons.shape
    tol = 0.01
    for expected_best_bs_value, best_bs_value, relative_time_horizon in zip(
        expected_best_bs_values, best_bs_values, relative_time_horizons
    ):
        assert (
            expected_best_bs_value <= best_bs_value.min() + tol
        ), relative_time_horizon

    # Check that the Brier score is discriminative enough to identify the true
    # survival probability by minimization.
    bs_min_estimates = survival_prob_grid[grid_bs_values.argmin(axis=0)]
    assert_allclose(bs_min_estimates, true_survival_at_horizons, atol=0.05)


@pytest.mark.parametrize("event_of_interest", [1, 2, 3])
@pytest.mark.parametrize("censored", [True, False])
@pytest.mark.parametrize("seed", range(2))
def test_brier_score_optimality_competing_risks(event_of_interest, censored, seed):
    # Check that IPCW brier_score_incidence behaves as a censor-agnostic proper
    # scoring rule for competing risks incidence probability estimation.
    n_samples = 8_000
    y, _, true_ci_funcs, t_max = _sample_test_data(
        n_samples=n_samples, censored=censored, random_state=seed
    )

    # Compute the theoretical cumulative incidence at given time horizons.
    relative_time_horizons = np.linspace(0, 1, num=11)
    time_horizons = relative_time_horizons * t_max
    true_incidence_at_horizons = true_ci_funcs[event_of_interest](time_horizons)

    # Check that our theoretical cumulative incidence estimate is compatible
    # with the Aalean-Johansen estimate. This only works for a large enough
    # value for n_samples.
    aj = AalenJohansenFitter(calculate_variance=False)
    aj.fit(
        durations=y["duration"],
        event_observed=y["event"],
        event_of_interest=event_of_interest,
    )
    aj_ci_df = aj.cumulative_density_
    aj_times = aj_ci_df.index
    aj_cumulative_incidence = aj_ci_df.values[:, 0]
    aj_ci_func = interp1d(
        aj_times,
        aj_cumulative_incidence,
        kind="previous",
        bounds_error=False,
        fill_value="extrapolate",
    )
    aj_incidence = aj_ci_func(time_horizons)

    # Decreasing atol requires increasing n_samples but would make the rest of
    # the test slower. Furthermore, we would expect the usual 1/sqrt(n_samples)
    # statistical convergence rate which is not very favorable.
    assert_allclose(aj_incidence, true_incidence_at_horizons, atol=0.03)

    # Compare the Brier score computed on the sampled events for the true
    # incidences at the given time horizons with the Brier scores computed on a
    # grid of candidate incidence values.
    expected_best_bs_values = brier_score_incidence(
        y,
        y,
        np.stack([true_incidence_at_horizons] * n_samples, axis=0),
        event_of_interest=event_of_interest,
        times=time_horizons,
    )

    incidence_grid = np.linspace(0, 1, num=21)
    grid_bs_values = np.stack(
        [
            brier_score_incidence(
                y,
                y,
                np.full(shape=(n_samples, time_horizons.size), fill_value=y_pred),
                event_of_interest=event_of_interest,
                times=time_horizons,
            )
            for y_pred in incidence_grid
        ],
        axis=0,
    )
    assert grid_bs_values.shape == (
        incidence_grid.size,
        relative_time_horizons.size,
    )
    # The true incidence should be the best one:
    best_bs_values = grid_bs_values.min(axis=0)
    assert best_bs_values.shape == time_horizons.shape
    assert expected_best_bs_values.shape == time_horizons.shape
    tol = 0.01
    for expected_best_bs_value, best_bs_value, relative_time_horizon in zip(
        expected_best_bs_values, best_bs_values, relative_time_horizons
    ):
        assert (
            expected_best_bs_value <= best_bs_value.min() + tol
        ), relative_time_horizon

    # Check that the Brier score is discriminative enough to identify the true
    # incidence by minimization.
    bs_min_estimates = incidence_grid[grid_bs_values.argmin(axis=0)]
    assert_allclose(bs_min_estimates, true_incidence_at_horizons, atol=0.05)
