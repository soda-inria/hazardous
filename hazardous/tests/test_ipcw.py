import numpy as np
import pytest
from lifelines.datasets import load_regression_dataset
from numpy.testing import assert_array_almost_equal

from .._ipcw import IpcwEstimator


@pytest.mark.parametrize("competitive_risk", [True, False])
def test_ipcw(competitive_risk):
    X = load_regression_dataset()
    y = dict(
        event=X["E"],
        duration=X["T"],
    )
    if competitive_risk:
        rng = np.random.default_rng(0)
        coef = rng.choice([1, 2], size=y["event"].shape[0])
        y["event"] *= coef

    times = np.arange(
        y["duration"].min(),
        y["duration"].max() - 1,
    )

    est = IpcwEstimator().fit(y)
    ipcw_probs = est.predict(times)

    expected_ipcw_probs = np.array(
        [
            1.0,
            1.0,
            1.0,
            1.01639935,
            1.02308619,
            1.03024064,
            1.03816557,
            1.03816557,
            1.06416786,
            1.06416786,
            1.08835349,
            1.08835349,
            1.08835349,
            1.19718884,
            1.33020982,
            1.33020982,
            1.33020982,
            1.33020982,
            1.33020982,
        ]
    )

    assert_array_almost_equal(ipcw_probs, expected_ipcw_probs)
