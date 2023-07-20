import re

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.model_selection import GridSearchCV, train_test_split

from hazardous import GradientBoostingIncidence
from hazardous.data import make_synthetic_competing_weibull
from hazardous.metrics import (
    integrated_brier_score_incidence,
    integrated_brier_score_survival,
)

# Change the following manually to check seed insensitivity:
SEED_RANGE = range(1)


@pytest.mark.parametrize("seed", SEED_RANGE)
def test_monotonic_gradient_boosting_incidence(seed):
    X, y = make_synthetic_competing_weibull(return_X_y=True, random_state=seed)
    assert sorted(y["event"].unique()) == [0, 1, 2, 3]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed)
    est = GradientBoostingIncidence(
        n_iter=3, event_of_interest="any", show_progressbar=False, random_state=seed
    )
    est.fit(X_train, y_train)
    assert_array_equal(est.event_ids_, [0, 1, 2, 3])

    survival_pred = est.predict_survival_function(X_test)
    assert survival_pred.shape == (X_test.shape[0], est.time_grid_.shape[0])
    assert np.all(survival_pred >= 0), survival_pred.min()
    assert np.all(survival_pred <= 1), survival_pred.max()

    ibs_gb_surv = integrated_brier_score_survival(
        y_train, y_test, survival_pred, times=est.time_grid_
    )
    assert 0 < ibs_gb_surv < 0.5, ibs_gb_surv

    # The score method of GradientBoostingIncidence with
    # event_of_interest="any" is the IBS of the survival function (or the IBS
    # of the cumulative incidence function for the any event boolean marker).
    assert ibs_gb_surv == pytest.approx(-est.score(X_test, y_test))

    # Check that the survival function is the complement of the cumulative
    # incidence function for any event.
    cif_pred = est.predict_cumulative_incidence(X_test)
    assert cif_pred.shape == (X_test.shape[0], est.time_grid_.shape[0])
    assert np.all(cif_pred >= 0), cif_pred.min()
    assert np.all(cif_pred <= 1), cif_pred.max()

    assert_allclose(survival_pred, 1 - cif_pred)
    ibs_gb_incidence = integrated_brier_score_incidence(
        y_train, y_test, cif_pred, times=est.time_grid_, event_of_interest="any"
    )
    assert ibs_gb_incidence == pytest.approx(ibs_gb_surv)


@pytest.mark.parametrize("seed", SEED_RANGE)
def test_gradient_boosting_any_event_survival(seed):
    X, y = make_synthetic_competing_weibull(return_X_y=True, random_state=seed)
    assert sorted(y["event"].unique()) == [0, 1, 2, 3]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed)
    est = GradientBoostingIncidence(
        n_iter=3, event_of_interest=3, show_progressbar=False, random_state=seed
    )
    est.fit(X_train, y_train)
    assert_array_equal(est.event_ids_, [0, 1, 2, 3])

    # Check that the survival function is the complement of the cumulative
    # incidence function for any event.
    cif_pred = est.predict_cumulative_incidence(X_test)
    assert cif_pred.shape == (X_test.shape[0], est.time_grid_.shape[0])
    assert np.all(cif_pred >= 0), cif_pred.min()
    assert np.all(cif_pred <= 1), cif_pred.max()

    ibs_gb_incidence = integrated_brier_score_incidence(
        y_train,
        y_test,
        cif_pred,
        times=est.time_grid_,
        event_of_interest=est.event_of_interest,
    )
    # The .score method of GradientBoostingIncidence with event_of_interest=3
    # is the IBS of the cumulative incidence function for the event of
    # interest.
    assert ibs_gb_incidence == pytest.approx(-est.score(X_test, y_test))

    # Asking to predict a cause-specific survival function for a specific event
    # of interest is mathematically possible but does not really make sense,
    # hence the warning.
    expected_msg = re.escape(
        "Values returned by predict_survival_function only make "
        "sense when the model is trained with a binary event "
        "indicator or when setting event_of_interest='any'. "
    )
    with pytest.warns(UserWarning, match=expected_msg):
        survival_pred = est.predict_survival_function(X_test)

    # Cause-specific "survival" is the complement of cause-specific CIF.
    assert_allclose(survival_pred, 1 - cif_pred)


@pytest.mark.parametrize("seed", SEED_RANGE)
def test_gradient_boosting_incidence_parameter_tuning(seed):
    # Minimal parameter grid with one poor and one good value for each
    # parameter to tune.
    param_grid = {
        "n_iter": [1, 10],
        "max_leaf_nodes": [2, 10],
        "hard_zero_fraction": [0.2, 1.0],
    }
    X, y = make_synthetic_competing_weibull(return_X_y=True, random_state=seed)
    assert sorted(y["event"].unique()) == [0, 1, 2, 3]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed)
    est = GradientBoostingIncidence(
        event_of_interest=3, show_progressbar=False, random_state=seed
    )
    grid_search = GridSearchCV(est, param_grid, cv=2, n_jobs=2)
    grid_search.fit(X_train, y_train)
    assert grid_search.best_params_ == {
        "n_iter": 10,
        "max_leaf_nodes": 10,
        "hard_zero_fraction": 0.2,
    }

    # Check that both the internal cross-validated IBS and the IBS on the test
    # set are good (lower IBS is better, hence higher negative IBS is better).
    max_expected_ibs = 0.17  # found emprically with different seed
    assert grid_search.best_score_ > -max_expected_ibs
    grid_search.best_estimator_.score(X_test, y_test) > -max_expected_ibs

    # Check that some other parameter values lead to much poorer IBS.
    cv_results = pd.DataFrame(grid_search.cv_results_).sort_values("mean_test_score")
    worst_ibs = -cv_results.iloc[0]["mean_test_score"]
    best_ibs = -cv_results.iloc[-1]["mean_test_score"]
    assert best_ibs == pytest.approx(-grid_search.best_score_)
    assert worst_ibs > 1.4 * best_ibs
