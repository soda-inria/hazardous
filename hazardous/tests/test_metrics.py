import re

import numpy as np
import pytest
from lifelines import CoxPHFitter
from lifelines.datasets import load_regression_dataset
from numpy.testing import assert_allclose

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
