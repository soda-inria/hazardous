from collections import Counter

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal, assert_equal

from hazardous.metrics._concordance_index import (
    _concordance_index_incidence_report,
    _concordance_index_tau,
    _concordance_summary_statistics,
    concordance_index_incidence,
    interpolate_preds,
)


@pytest.mark.parametrize(
    "bunch, expected",
    [
        # Only 3 concordant pairs over 5 comparable pairs
        (
            dict(
                event=np.array([1, 1, 0, 0]),
                duration=np.array([10, 20, 30, 40]),
                y_pred=np.array([0.9, 0.1, 0.2, 0.3]),
                ipcw=np.array([1, 1, 1, 1]),
            ),
            Counter(
                dict(
                    num_pairs=5,
                    weighted_pairs=5,
                    num_concordant_pairs=3,
                    weighted_concordant_pairs=3,
                )
            ),
        ),
        # 0 concordant pairs over 5 comparable pairs
        (
            dict(
                event=np.array([1, 1, 0, 0]),
                duration=np.array([10, 20, 30, 40]),
                y_pred=np.array([0.1, 0.15, 0.2, 0.3]),
                ipcw=np.array([1, 1, 1, 1]),
            ),
            Counter(
                dict(
                    num_pairs=5,
                    weighted_pairs=5,
                )
            ),
        ),
        # No comparable pairs at all
        (
            dict(
                event=np.array([0, 0, 0, 0]),
                duration=np.array([10, 20, 30, 40]),
                y_pred=np.array([0.1, 0.15, 0.2, 0.3]),
                ipcw=np.array([1, 1, 1, 1]),
            ),
            Counter(),
        ),
        # No censored pairs
        (
            dict(
                event=np.array([1, 1, 1, 1]),
                duration=np.array([10, 20, 30, 40]),
                y_pred=np.array([0.9, 0.8, 0.7, 0.6]),
                ipcw=np.array([1, 1, 1, 1]),
            ),
            Counter(
                dict(
                    num_pairs=6,
                    weighted_pairs=6.0,
                    num_concordant_pairs=6,
                    weighted_concordant_pairs=6.0,
                )
            ),
        ),
        # All pairs are concordant, and the first individual has a weight != 1
        (
            dict(
                event=np.array([1, 1, 0, 0]),
                duration=np.array([10, 20, 30, 40]),
                y_pred=np.array([0.9, 0.8, 0.2, 0.3]),
                ipcw=np.array([0.5, 1, 1, 1]),
            ),
            Counter(
                dict(
                    weighted_pairs=14.0,  # 4 + (4 + 1) + (4 + 1)
                    weighted_concordant_pairs=14.0,
                    num_pairs=5,
                    num_concordant_pairs=5,
                )
            ),
        ),
    ],
)
def test_summary_statistics_a(bunch, expected):
    stats_a = _concordance_summary_statistics(
        pair_type="a",
        **bunch,
    )
    assert stats_a == expected


@pytest.mark.parametrize(
    "bunch, expected",
    [
        # 0 comparable pairs, because we only accepts pairs where D_i = 1 and D_j = 2
        (
            dict(
                event=np.array([1, 1, 1, 1]),
                duration=np.array([10, 20, 30, 40]),
                y_pred=np.array([0.9, 0.8, 0.7, 0.6]),
                ipcw=np.array([1, 1, 1, 1]),
            ),
            Counter(),
        ),
        (
            # 0 comparable pairs, because we only accepts pairs where
            # D_i = 1 and D_j = 2
            dict(
                event=np.array([0, 0, 0, 0]),
                duration=np.array([10, 20, 30, 40]),
                y_pred=np.array([0.9, 0.8, 0.7, 0.6]),
                ipcw=np.array([1, 1, 1, 1]),
            ),
            Counter(),
        ),
        # 0 comparable pairs, because we only accept pairs where T_i >= T_j
        (
            dict(
                event=np.array([1, 1, 0, 0]),
                duration=np.array([10, 20, 30, 40]),
                y_pred=np.array([0.9, 0.8, 0.7, 0.6]),
                ipcw=np.array([1, 1, 1, 1]),
            ),
            Counter(),
        ),
        # 4 concordant pairs out of 4 comparable pairs
        (
            dict(
                event=np.array([1, 1, 0, 0]),
                duration=np.array([30, 40, 10, 20]),
                y_pred=np.array([0.9, 0.8, 0.7, 0.6]),
                ipcw=np.array([1, 1, 1, 1]),
            ),
            Counter(
                dict(
                    num_pairs=4,
                    weighted_pairs=4.0,
                    num_concordant_pairs=4,
                    weighted_concordant_pairs=4.0,
                )
            ),
        ),
        # 1 concordant pairs out of 2 comparable pairs
        (
            dict(
                event=np.array([1, 1, 0, 0]),
                duration=np.array([30, 5, 10, 20]),
                y_pred=np.array([0.9, 0.8, 1, 0.6]),
                ipcw=np.array([1, 1, 1, 1]),
            ),
            Counter(
                dict(
                    num_pairs=2,
                    weighted_pairs=2.0,
                    num_concordant_pairs=1,
                    weighted_concordant_pairs=1.0,
                )
            ),
        ),
        # 2 concordant pairs out of 2 with weight != 1
        (
            dict(
                event=np.array([1, 1, 0, 0]),
                duration=np.array([30, 5, 10, 20]),
                y_pred=np.array([0.9, 0.8, 0.4, 0.6]),
                ipcw=np.array([0.1, 1, 1, 0.5]),
            ),
            Counter(
                dict(
                    num_pairs=2,
                    weighted_pairs=30.0,  # (1/0.1 + 1/0.1 * 1/0.5)
                    num_concordant_pairs=2,
                    weighted_concordant_pairs=30.0,
                )
            ),
        ),
        # 4 concordant pairs out of 4 with weight != 1 and ties on duration
        (
            dict(
                event=np.array([1, 1, 0, 0]),
                duration=np.array([30, 30, 10, 20]),
                y_pred=np.array([0.9, 0.8, 0.4, 0.6]),
                ipcw=np.array([0.1, 1, 1, 0.5]),
            ),
            Counter(
                dict(
                    weighted_pairs=33.0,  # (1/0.1 + 1/0.1 * 1/0.5 + 1 + 1/0.5)
                    weighted_concordant_pairs=33.0,
                    num_pairs=4,
                    num_concordant_pairs=4,
                    num_ties_times=1,
                )
            ),
        ),
        # 2 concordant pairs out of 4 comparable pairs, 2 prediction ties
        (
            dict(
                event=np.array([1, 1, 0, 0]),
                duration=np.array([40, 30, 10, 20]),
                y_pred=np.array([0.9, 0.9, 0.9, 0.6]),
                ipcw=np.array([0.1, 1, 1, 0.5]),
            ),
            Counter(
                dict(
                    weighted_pairs=33.0,
                    weighted_concordant_pairs=22.0,
                    weighted_ties_pred=11.0,
                    num_pairs=4,
                    num_concordant_pairs=2,
                    num_ties_pred=2,
                )
            ),
        ),
    ],
)
def test_summary_statistics_b(bunch, expected):
    stats_b = _concordance_summary_statistics(
        pair_type="b",
        **bunch,
    )
    assert stats_b == expected


@pytest.mark.parametrize(
    "bunch, expected",
    [
        # 5 concordant pairs out of 5 comparable pairs, no competing event
        (
            dict(
                y_test=pd.DataFrame(
                    dict(
                        event=np.array([1, 1, 0, 0]),
                        duration=np.array([10, 20, 30, 40]),
                    )
                ),
                y_pred=np.array([0.9, 0.8, 0.7, 0.6]),
                ipcw=np.array([1, 1, 1, 1]),
            ),
            dict(
                cindex=1.0,
                num_pairs_a=5,
                num_concordant_pairs_a=5,
                num_ties_pred_a=0,
                num_ties_times_a=0,
                num_pairs_b=0,
                num_concordant_pairs_b=0,
                num_ties_pred_b=0,
            ),
        ),
        # 3 concordant pairs out of 3 comparable pairs, with competing event
        (
            dict(
                y_test=pd.DataFrame(
                    dict(
                        event=np.array([2, 1, 0, 0]),
                        duration=np.array([10, 20, 30, 40]),
                    )
                ),
                y_pred=np.array([0.8, 0.9, 0.7, 0.6]),
                ipcw=np.array([1, 1, 1, 1]),
            ),
            dict(
                cindex=1.0,
                num_pairs_a=2,
                num_concordant_pairs_a=2,
                num_ties_pred_a=0,
                num_ties_times_a=0,
                num_pairs_b=1,
                num_concordant_pairs_b=1,
                num_ties_pred_b=0,
            ),
        ),
        # ties on duration for the event of interest
        (
            dict(
                y_test=pd.DataFrame(
                    dict(
                        event=np.array([1, 1, 0, 0]),
                        duration=np.array([10, 10, 30, 40]),
                    )
                ),
                y_pred=np.array([0.9, 0.8, 0.7, 0.6]),
                ipcw=np.array([1, 1, 1, 1]),
            ),
            dict(
                cindex=1.0,
                num_pairs_a=4,
                num_concordant_pairs_a=4,
                num_ties_pred_a=0,
                num_ties_times_a=1,
                num_pairs_b=0,
                num_concordant_pairs_b=0,
                num_ties_pred_b=0,
            ),
        ),
    ],
)
def test_cindex_tau(bunch, expected):
    stats = _concordance_index_tau(event_of_interest=1, **bunch)
    assert stats == expected


def test_cindex_tau_no_comparable_pairs():
    bunch = dict(
        y_test=pd.DataFrame(
            dict(
                event=np.array([1, 0, 0]),
                duration=np.array([30, 20, 10]),
            )
        ),
        y_pred=np.array([0.9, 0.7, 0.6]),
        ipcw=np.array([1, 1, 1]),
    )
    # No comparable pairs because the event of interest is not present
    msg = r"There is not any event for event_of_interest=2."
    with pytest.warns(UserWarning, match=msg):
        stats = _concordance_index_tau(event_of_interest=2, **bunch)
        assert_equal(stats["cindex"], np.nan)

    # No comparable pairs because T_i > T_j
    stats = _concordance_index_tau(event_of_interest=1, **bunch)
    assert_equal(stats["cindex"], np.nan)

    # No comparable pairs because the event of interest is not present
    bunch["y_test"]["event"] = pd.Series([0, 0, 0])
    msg = r"There is not any event for event_of_interest=1."
    with pytest.warns(UserWarning, match=msg):
        stats = _concordance_index_tau(event_of_interest=1, **bunch)
        assert_equal(stats["cindex"], np.nan)


def test_concordance_index_incidence_report():
    y_test = pd.DataFrame(
        dict(
            event=[1, 0, 0],
            duration=[20, 30, 40],
        )
    )
    y_pred = np.array(
        [
            [0.9, 0.7, 0.3],
            [0.8, 0.6, 0.2],
            [0.7, 0.4, 0.1],
        ]
    )
    time_grid = [10, 20, 30]
    taus = [15, 25]

    msg = "There is not any event for event_of_interest=1"
    with pytest.warns(UserWarning, match=msg):
        stats = _concordance_index_incidence_report(
            y_test,
            y_pred,
            ipcw_estimator=None,
            time_grid=time_grid,
            taus=taus,
            event_of_interest=1,
        )
    expected = dict(
        cindex=[np.nan, 1.0],
        num_pairs_a=[0, 2],
        num_concordant_pairs_a=[0, 2],
        num_ties_pred_a=[0, 0],
        num_ties_times_a=[0, 0],
        num_pairs_b=[0, 0],
        num_concordant_pairs_b=[0, 0],
        num_ties_pred_b=[0, 0],
    )
    assert_equal(stats, expected)


def test_concordance_index_incidence_report_incorrect_input():
    # y_train shouldn't be passed when ipcw_estimator is None
    y_test = pd.DataFrame(
        dict(
            event=[1, 0, 0],
            duration=[20, 30, 40],
        )
    )
    y_pred = np.array(
        [
            [0.9],
            [0.8],
            [0.7],
        ]
    )
    y_train = y_test.copy()

    msg = "y_train passed but ipcw_estimator is set to None"
    with pytest.warns(UserWarning, match=msg):
        _concordance_index_incidence_report(
            y_test,
            y_pred,
            ipcw_estimator=None,
            y_train=y_train,
        )

    msg = "ipcw_estimator is set, but y_train is None"
    with pytest.raises(ValueError, match=msg):
        _concordance_index_incidence_report(
            y_test,
            y_pred,
            ipcw_estimator="km",  # default
            y_train=None,  # default
        )

    # When taus is passed, time_grid must be set
    msg = "When 'taus' is set, 'time_grid' must also be set"
    with pytest.raises(ValueError, match=msg):
        _concordance_index_incidence_report(
            y_test,
            y_pred,
            ipcw_estimator=None,
            y_train=y_train,
            taus=[20],
        )


def test_concordance_index_incidence():
    y_test = pd.DataFrame(
        dict(
            event=[1, 0, 0],
            duration=[20, 30, 40],
        )
    )
    y_pred = np.array(
        [
            [0.5, 0.7, 0.9],
            [0.3, 0.6, 0.8],
            [0.2, 0.3, 0.4],
        ]
    )
    time_grid = [10, 20, 30]
    taus = [15, 25]
    cindex = concordance_index_incidence(
        y_test,
        y_pred,
        y_train=y_test.copy(),
        time_grid=time_grid,
        taus=taus,
        event_of_interest=1,
        ipcw_estimator="km",
    )
    expected = [np.nan, 1]
    assert_array_equal(cindex, expected)


@pytest.mark.parametrize(
    "bunch, expected",
    [
        # When tau is higher than the max of the time_grid,
        # we select the last column of y_pred.
        (
            dict(
                y_pred=np.array(
                    [
                        [0.5, 0.7, 0.9],
                        [0.3, 0.6, 0.8],
                        [0.2, 0.3, 0.4],
                    ]
                ),
                time_grid=[10, 20, 30],
                tau=50,
            ),
            np.array([0.9, 0.8, 0.4]),
        ),
        # When tau is lower than the min of the time_grid,
        # we select the first column of y_pred
        (
            dict(
                y_pred=np.array(
                    [
                        [0.5, 0.7, 0.9],
                        [0.3, 0.6, 0.8],
                        [0.2, 0.3, 0.4],
                    ]
                ),
                time_grid=[10, 20, 30],
                tau=5,
            ),
            np.array([0.5, 0.3, 0.2]),
        ),
        # When y_pred has a single column, tau has no effect
        (
            dict(
                y_pred=np.array(
                    [
                        [0.9],
                        [0.8],
                        [0.4],
                    ]
                ),
                time_grid=[10],
                tau=5,
            ),
            np.array([0.9, 0.8, 0.4]),
        ),
        # Otherwise, apply linear interpolation
        (
            dict(
                y_pred=np.array(
                    [
                        [0.5, 0.7, 0.9],
                        [0.3, 0.6, 0.8],
                        [0.2, 0.3, 0.4],
                    ]
                ),
                time_grid=[10, 20, 30],
                tau=25,
            ),
            np.array([0.8, 0.7, 0.35]),
        ),
    ],
)
def test_interpolate_preds(bunch, expected):
    y_pred_interp = interpolate_preds(**bunch)
    assert_array_almost_equal(y_pred_interp, expected)
