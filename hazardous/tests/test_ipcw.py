import numpy as np
import pytest
from lifelines.datasets import load_regression_dataset
from numpy.testing import assert_allclose

from .._ipcw import IPCWEstimator


@pytest.mark.parametrize("competing_risk", [True, False])
def test_ipcw_consistent_with_sksurv(competing_risk):
    data = load_regression_dataset()
    y = dict(
        event=data["E"],
        duration=data["T"],
    )
    if competing_risk:
        rng = np.random.default_rng(0)
        event_id_modifier = rng.choice([1, 2], size=y["event"].shape[0])
        y["event"] *= event_id_modifier

    times = np.arange(
        y["duration"].min(),
        y["duration"].max() - 1,
    )

    est = IPCWEstimator().fit(y)
    ipcw_values = est.compute_ipcw_at(times)

    # Expected values computed with scikit-survival:
    #
    # from sksurv.nonparametric import CensoringDistributionEstimator
    # from hazardous.utils import _dict_to_recarray
    # from pprint import pprint
    # cens = CensoringDistributionEstimator()
    # y_bool_event = {
    #     "duration": y["duration"],
    #     "event": (y["event"] != 0).astype(bool),
    # }
    # cens.fit(_dict_to_recarray(y_bool_event))
    # y_test = _dict_to_recarray({
    #     "duration": times,
    #     "event": np.ones_like(times).astype(bool),
    # })
    # pprint(cens.predict_ipcw(y_test).tolist())

    expected_ipcw_values = np.array(
        [
            1.0,
            1.0,
            1.0,
            1.0163993512344567,
            1.0230861890715255,
            1.0302406379461515,
            1.0381655659303526,
            1.0381655659303526,
            1.0641678574703521,
            1.0641678574703521,
            1.0883534905946783,
            1.0883534905946783,
            1.0883534905946783,
            1.1971888396541464,
            1.3302098218379401,
            1.3302098218379401,
            1.3302098218379401,
            1.3302098218379401,
            1.3302098218379401,
        ]
    )
    assert_allclose(ipcw_values, expected_ipcw_values)

    # Remove censoring: the weights should be 1.
    y_uncensored = y.copy()
    if competing_risk:
        y_uncensored["event"] = rng.choice([1, 2], size=y["event"].shape[0])
    else:
        y_uncensored["event"] = np.ones_like(y["event"])
    est = IPCWEstimator().fit(y_uncensored)
    ipcw_values = est.compute_ipcw_at(times)
    assert_allclose(ipcw_values, np.ones_like(ipcw_values))
