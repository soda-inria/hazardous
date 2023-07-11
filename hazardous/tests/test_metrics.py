import re

import numpy as np
import pandas as pd
import pytest
from lifelines import CoxPHFitter
from lifelines.datasets import load_regression_dataset
from numpy.testing import assert_allclose, assert_array_equal

from ..metrics import (
    brier_score_incidence,
    brier_score_survival,
    integrated_brier_score_incidence,
    integrated_brier_score_survival,
)

X = load_regression_dataset()
X_train, X_test = X.iloc[:150], X.iloc[150:]
y_train = dict(
    event=X_train["E"],
    duration=X_train["T"],
)
y_test = dict(
    event=X_test["E"],
    duration=X_test["T"],
)
times = np.arange(
    y_test["duration"].min(),
    y_test["duration"].max() - 1,
)

est = CoxPHFitter().fit(X_train, duration_col="T", event_col="E")
y_pred = est.predict_survival_function(X_test, times)
y_pred = y_pred.T.values  # (n_samples, n_times)


@pytest.mark.parametrize("event_of_interest", [1, "any"])
def test_brier_score_computer(event_of_interest):
    times_, loss = brier_score_survival(
        y_train,
        y_test,
        y_pred,
        times,
        event_of_interest,
    )

    # Check that 'times_' hasn't been changed
    assert_array_equal(times, times_)

    loss_expected = np.array(
        [
            0.01921016,
            0.08987548,
            0.11693115,
            0.18832202,
            0.21346599,
            0.24300206,
            0.24217776,
            0.21987924,
            0.19987174,
            0.16301318,
            0.07628881,
            0.05829176,
            0.0663998,
            0.04524901,
            0.04553689,
            0.02250038,
            0.02259133,
        ]
    )
    assert_allclose(loss, loss_expected, atol=1e-6)

    ibs = integrated_brier_score_survival(
        y_train,
        y_test,
        y_pred,
        times,
        event_of_interest,
    )

    ibs_expected = 0.1257316251344779
    assert ibs == pytest.approx(ibs_expected, abs=1e-6)


@pytest.mark.parametrize("event_of_interest", [1, "any"])
def test_brier_score_incidence_computer(event_of_interest):
    times_, loss = brier_score_survival(
        y_train,
        y_test,
        y_pred,
        times,
        event_of_interest,
    )
    times_incidence_, loss_incidence = brier_score_incidence(
        y_train,
        y_test,
        1 - y_pred,
        times,
        event_of_interest,
    )

    assert_array_equal(times_, times_incidence_)
    assert_array_equal(loss, loss_incidence)

    ibs = integrated_brier_score_survival(
        y_train,
        y_test,
        y_pred,
        times,
        event_of_interest,
    )
    ibs_incidence = integrated_brier_score_incidence(
        y_train,
        y_test,
        1 - y_pred,
        times,
        event_of_interest,
    )

    assert ibs == ibs_incidence


def test_brier_score_warnings_on_competive_event():
    coef = np.random.choice([1, 2], size=y_train["event"].shape[0])
    y_train["event"] *= coef

    msg = "Computing the Brier Score only make sense"
    with pytest.warns(match=msg):
        brier_score_survival(
            y_train,
            y_test,
            y_pred,
            times,
            event_of_interest=2,
        )

    with pytest.warns(None):
        brier_score_incidence(
            y_train,
            y_test,
            1 - y_pred,
            times,
            event_of_interest=2,
        )


def test_brier_score_incidence_warnings_surv_input():
    msg = "\n\nThe average shape of the y_pred curve is decreasing."
    with pytest.warns(match=re.escape(msg)):
        brier_score_incidence(
            y_train,
            y_test,
            y_pred,
            times,
            event_of_interest="any",
        )

    with pytest.warns(None):
        brier_score_survival(
            y_train,
            y_test,
            y_pred,
            times,
            event_of_interest="any",
        )


def test_brier_score_survival_wrong_parameters():
    msg = "event_of_interest must be a strictly positive integer or 'any'"
    for event_of_interest in [-10, 0, "wrong_event"]:
        with pytest.raises(ValueError, match=msg):
            brier_score_survival(
                y_train,
                y_test,
                y_pred,
                times,
                event_of_interest,
            )

    msg = "event_of_interest must be an instance of"
    for event_of_interest in [None, [1], (2, 3)]:
        with pytest.raises(TypeError, match=msg):
            brier_score_survival(
                y_train,
                y_test,
                y_pred,
                times,
                event_of_interest,
            )


def _dict_to_pd(y):
    return pd.DataFrame(y)


def _dict_to_recarray(y):
    y_out = np.empty(
        shape=y["event"].shape[0],
        dtype=[("event", np.int32), ("duration", np.float64)],
    )
    y_out["event"] = y["event"]
    y_out["duration"] = y["duration"]
    return y_out


@pytest.mark.parametrize("format_func", [_dict_to_pd, _dict_to_recarray])
def test_test_brier_score_survival_inputs_format(format_func):
    _, loss = brier_score_survival(
        format_func(y_train),
        format_func(y_test),
        y_pred,
        times,
        event_of_interest="any",
    )

    loss_expected = np.array(
        [
            0.01921016,
            0.08987548,
            0.11693115,
            0.18832202,
            0.21346599,
            0.24300206,
            0.24217776,
            0.21987924,
            0.19987174,
            0.16301318,
            0.07628881,
            0.05829176,
            0.0663998,
            0.04524901,
            0.04553689,
            0.02250038,
            0.02259133,
        ]
    )
    assert_allclose(loss, loss_expected, atol=1e-6)


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
            y_pred,
            times,
            event_of_interest="any",
        )

    msg = "'times' length (5) must be equal to y_pred.shape[1] (17)."
    with pytest.raises(ValueError, match=re.escape(msg)):
        brier_score_survival(
            y_train,
            y_test,
            y_pred,
            times[:5],
            event_of_interest="any",
        )
