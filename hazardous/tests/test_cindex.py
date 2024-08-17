from collections import Counter

import numpy as np
import pandas as pd
import pytest
from lifelines import CoxPHFitter
from lifelines.datasets import load_kidney_transplant
from numpy.testing import (
    assert_almost_equal,
    assert_array_almost_equal,
    assert_array_equal,
    assert_equal,
)
from sklearn.model_selection import train_test_split

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
                    n_pairs=5,
                    weighted_pairs=5,
                    n_concordant_pairs=3,
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
                    n_pairs=5,
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
                    n_pairs=6,
                    weighted_pairs=6.0,
                    n_concordant_pairs=6,
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
                ipcw=np.array([1 / 0.5, 1, 1, 1]),
            ),
            Counter(
                dict(
                    weighted_pairs=14.0,  # 3 * 4 + 2 * 1
                    weighted_concordant_pairs=14.0,
                    n_pairs=5,
                    n_concordant_pairs=5,
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
                    n_pairs=4,
                    weighted_pairs=4.0,
                    n_concordant_pairs=4,
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
                    n_pairs=2,
                    weighted_pairs=2.0,
                    n_concordant_pairs=1,
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
                ipcw=np.array([1 / 0.1, 1, 1, 1 / 0.5]),
            ),
            Counter(
                dict(
                    n_pairs=2,
                    weighted_pairs=30.0,  # (1/0.1 + 1/0.1 * 1/0.5)
                    n_concordant_pairs=2,
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
                ipcw=np.array([1 / 0.1, 1, 1, 1 / 0.5]),
            ),
            Counter(
                dict(
                    weighted_pairs=33.0,  # (1/0.1 + 1/0.1 * 1/0.5 + 1 + 1/0.5)
                    weighted_concordant_pairs=33.0,
                    n_pairs=4,
                    n_concordant_pairs=4,
                    n_ties_times=1,
                )
            ),
        ),
        # 2 concordant pairs out of 4 comparable pairs, 2 prediction ties
        (
            dict(
                event=np.array([1, 1, 0, 0]),
                duration=np.array([40, 30, 10, 20]),
                y_pred=np.array([0.9, 0.9, 0.9, 0.6]),
                ipcw=np.array([1 / 0.1, 1, 1, 1 / 0.5]),
            ),
            Counter(
                dict(
                    weighted_pairs=33.0,
                    weighted_concordant_pairs=22.0,
                    weighted_ties_pred=11.0,
                    n_pairs=4,
                    n_concordant_pairs=2,
                    n_ties_pred=2,
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
                n_pairs_a=5,
                n_concordant_pairs_a=5,
                n_ties_pred_a=0,
                n_ties_times_a=0,
                n_pairs_b=0,
                n_concordant_pairs_b=0,
                n_ties_pred_b=0,
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
                n_pairs_a=2,
                n_concordant_pairs_a=2,
                n_ties_pred_a=0,
                n_ties_times_a=0,
                n_pairs_b=1,
                n_concordant_pairs_b=1,
                n_ties_pred_b=0,
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
                n_pairs_a=4,
                n_concordant_pairs_a=4,
                n_ties_pred_a=0,
                n_ties_times_a=1,
                n_pairs_b=0,
                n_concordant_pairs_b=0,
                n_ties_pred_b=0,
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
        taus=taus,
        cindex=[np.nan, 1.0],
        n_pairs_a=[0, 2],
        n_concordant_pairs_a=[0, 2],
        n_ties_pred_a=[0, 0],
        n_ties_times_a=[0, 0],
        n_pairs_b=[0, 0],
        n_concordant_pairs_b=[0, 0],
        n_ties_pred_b=[0, 0],
    )
    assert_equal(stats, expected)


def test_concordance_index_incidence_report_competitive():
    y_test = pd.DataFrame(
        dict(
            event=[1, 2, 0, 0],
            duration=[10, 20, 30, 40],
        )
    )
    y_pred = np.array(
        [
            [0.9, 0.7, 0.3],
            [0.8, 0.6, 0.2],
            [0.7, 0.4, 0.1],
            [0.3, 0.2, 0.1],
        ]
    )
    time_grid = [10, 20, 30]

    res = _concordance_index_incidence_report(
        y_test, y_pred, time_grid, taus=None, y_train=y_test, event_of_interest=1
    )
    assert res == {
        "cindex": [1.0],
        "n_concordant_pairs_a": [3],
        "n_concordant_pairs_b": [0],
        "n_pairs_a": [3],
        "n_pairs_b": [0],
        "n_ties_pred_a": [0],
        "n_ties_pred_b": [0],
        "n_ties_times_a": [0],
        "taus": np.array([40]),
    }

    res = _concordance_index_incidence_report(
        y_test, y_pred, time_grid, taus=None, y_train=y_test, event_of_interest=2
    )

    assert res == {
        "cindex": [0.6666666666666666],
        "n_concordant_pairs_a": [2],
        "n_concordant_pairs_b": [0],
        "n_pairs_a": [2],
        "n_pairs_b": [1],
        "n_ties_pred_a": [0],
        "n_ties_pred_b": [0],
        "n_ties_times_a": [0],
        "taus": np.array([40]),
    }


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


def test_sksurv_consistency_synthetic():
    y_test = pd.DataFrame(
        dict(
            duration=[10, 20, 30, 40],
            event=[1, 1, 0, 0],
        )
    )
    y_pred = np.array(
        [
            [0.8, 0.82, 0.84],
            [0.2, 0.22, 0.24],
            [0.3, 0.32, 0.34],
            [0.3, 0.32, 0.34],
        ]
    )
    time_grid = [15, 25, 35]

    # First, test without tau
    result = _concordance_index_incidence_report(
        y_test=y_test,
        y_pred=y_pred,
        time_grid=time_grid,
        taus=None,
        y_train=y_test,
    )

    # from sksurv.metrics import concordance_index_ipcw
    # concordance_index_ipcw(
    #     make_recarray(y_test),
    #     make_recarray(y_test),
    #     y_pred[:, -1],
    #     tau=None,
    # )
    sksurv_result = [0.6, 3, 2, 0, 0]
    metric_names = [
        "cindex",
        "n_concordant_pairs_a",
        "n_pairs_a",
        "n_ties_pred_a",
        "n_ties_times_a",
    ]
    sksurv_result = dict(zip(metric_names, sksurv_result))
    sksurv_result["n_pairs_a"] += sksurv_result["n_concordant_pairs_a"]

    for metric in metric_names:
        assert sksurv_result[metric] == result[metric][0]

    # Next, test with some tau between t_min and t_max
    result = _concordance_index_incidence_report(
        y_test=y_test,
        y_pred=y_pred,
        time_grid=time_grid,
        taus=[15],
        y_train=y_test,
    )
    # concordance_index_ipcw(
    #     make_recarray(y_test),
    #     make_recarray(y_test),
    #     y_pred[:, 0],
    #     tau=15,
    # )
    #
    # Note that sksurv is not consistent between
    # the cindex and the number of concordant & discordant.
    # In sksurv, the number of concordant and discordant are invariant of taus.
    # With our approach, they should be respectively 3 and 0.
    # Therefore, we only compare the cindex.
    sksurv_result = [1, 3, 2, 0, 0]
    sksurv_result = dict(zip(metric_names, sksurv_result))

    assert sksurv_result["cindex"] == result["cindex"][0]


@pytest.mark.parametrize(
    "random_state, sksurv_cindex",
    [
        (0, 0.77798176),
        (1, 0.69160605),
        (2, 0.68663299),
        (3, 0.69568347),
        (4, 0.64405581),
    ],
)
def test_sksurv_consistancy_kidney(random_state, sksurv_cindex):
    df = load_kidney_transplant()
    df_train, df_test = train_test_split(
        df, stratify=df["death"], random_state=random_state
    )
    cox = CoxPHFitter().fit(df_train, duration_col="time", event_col="death")

    t_min, t_max = df["time"].min(), df["time"].max()
    time_grid = np.linspace(t_min, t_max, 20)
    y_pred = 1 - cox.predict_survival_function(df_test, times=time_grid).T.to_numpy()

    y_test = df_test[["death", "time"]].rename(
        columns=dict(death="event", time="duration")
    )
    y_train = df_train[["death", "time"]].rename(
        columns=dict(death="event", time="duration")
    )

    result = _concordance_index_incidence_report(
        y_test=y_test,
        y_pred=y_pred,
        time_grid=time_grid,
        taus=None,
        y_train=y_train,
    )

    # concordance_index_ipcw(
    #     make_recarray(y_train),
    #     make_recarray(y_test),
    #     y_pred[:, -1],
    #     tau=None,
    # )[0]
    assert_almost_equal(result["cindex"][0], sksurv_cindex, decimal=4)
