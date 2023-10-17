import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from pandas.testing import assert_frame_equal

from hazardous.survtrace._encoder import SurvFeatureEncoder, SurvTargetEncoder

X = pd.DataFrame(
    dict(
        a=[1, 2, 3],
        b=["a", "b", "a"],
        c=["c", "d", "e"],
    )
)

y = pd.DataFrame(
    dict(
        event=[1.0, 0.0, 1.0, 0.0, 0.0],
        duration=[81, 7, 28, 75, 57],
    )
)


def test_surv_feature_encoder():
    f_enc = SurvFeatureEncoder().fit(X)

    expected_vocab_size = np.array([0, 2])
    assert_array_equal(f_enc.vocab_size_, expected_vocab_size)

    expected_num_cols = ["a"]
    expected_categ_cols = ["b", "c"]

    assert_array_equal(f_enc.categ_cols_, expected_categ_cols)
    assert_array_equal(f_enc.num_cols_, expected_num_cols)

    X_trans = f_enc.transform(X)
    expected_X_trans = np.array(
        [
            [0.0, 2.0, -1.22474487],
            [1.0, 3.0, 0.0],
            [0.0, 4.0, 1.22474487],
        ]
    )
    assert_array_almost_equal(X_trans, expected_X_trans)


def test_surv_feature_passing_cols():
    f_enc = SurvFeatureEncoder(num_cols="a", categ_cols=["b", "c"])
    f_enc.fit(X)

    # check original input unchanged
    assert_array_equal(f_enc.categ_cols, ["b", "c"])
    assert_array_equal(f_enc.num_cols, "a")

    # check new input
    assert_array_equal(f_enc.categ_cols_, ["b", "c"])
    assert_array_equal(f_enc.num_cols_, ["a"])


def test_surv_feature_bad_input():
    with pytest.raises(TypeError, match=r"(?=.*dataframe)"):
        SurvFeatureEncoder().fit(X.values)


def test_surv_target():
    t_enc = SurvTargetEncoder().fit(y)

    expected_quantile_horizons = [0.25, 0.5, 0.75]
    expected_time_grid = np.array([0.0, 41.25, 54.5, 67.75, 81.0])
    expected_time_grid_to_idx = dict(
        zip(expected_time_grid, range(len(expected_time_grid)))
    )

    assert t_enc.quantile_horizons_ == expected_quantile_horizons
    assert_array_equal(t_enc.time_grid_, expected_time_grid)
    assert t_enc.time_grid_to_idx_ == expected_time_grid_to_idx

    y_trans = t_enc.transform(y)

    event = y["event"].astype("int64")
    duration = [3, 0, 0, 3, 2]
    frac_duration = np.array(
        [1.0, 0.16969697, 0.6787879, 0.5471698, 0.18867925]
    ).astype("float32")
    expected_y_trans = pd.DataFrame(
        dict(
            event=event,
            duration=duration,
            frac_duration=frac_duration,
        )
    )
    assert_frame_equal(y_trans, expected_y_trans)
