import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal

from hazardous.metrics import accuracy_in_time


@pytest.mark.parametrize("quantiles", [None, np.array([0.1, 0.4, 0.6, 0.8])])
def test_accuracy_in_time_basic(quantiles):
    """Test basic functionality."""
    y_test = pd.DataFrame({"event": [1, 0, 1], "duration": [5, 10, 15]})
    y_pred = np.array(
        [
            [[0.2, 0.5], [0.8, 0.5]],  # Sample 1
            [[0.7, 0.6], [0.3, 0.4]],  # Sample 2
            [[0.1, 0.6], [0.9, 0.4]],  # Sample 3
        ]
    )
    # t = 5: (1, 2) correct = 2/3
    # t = 10: (1) is a tie, incorrect by luck, (2) doesn't count and (3) correct = 1/2
    expected_acc_in_time = np.array([2 / 3, 1 / 2])

    time_grid = np.array([5, 10])
    expected_taus = time_grid

    # For small time grid and large quantiles, the acc in time is invariant to quantiles
    acc_in_time, taus = accuracy_in_time(y_test, y_pred, time_grid, quantiles=quantiles)
    assert_array_equal(expected_acc_in_time, acc_in_time)
    assert_array_equal(expected_taus, taus)


def test_invalid_y_pred_shape():
    """Test invalid y_pred shape."""
    y_test = pd.DataFrame({"event": [1, 0, 2], "duration": [5, 10, 15]})
    y_pred = np.array([[0.2, 0.5], [0.8, 0.5]])  # Incorrect shape
    time_grid = np.array([5, 10])

    with pytest.raises(ValueError, match="'y_pred' must be a 3D array"):
        accuracy_in_time(y_test, y_pred, time_grid)


def test_invalid_time_grid_length():
    """Test time grid length mismatch."""
    y_test = pd.DataFrame({"event": [1, 0, 2], "duration": [5, 10, 15]})
    y_pred = np.array(
        [
            [[0.2, 0.5], [0.8, 0.5]],
            [[0.7, 0.6], [0.3, 0.4]],
            [[0.1, 0.4], [0.9, 0.6]],
        ]
    )
    time_grid = np.array([5])  # Length mismatch

    with pytest.raises(ValueError, match="'time_grid' length"):
        accuracy_in_time(y_test, y_pred, time_grid)


def test_same_number_of_samples():
    """Test y_test and y_pred first dimension mismatch"""
    y_test = pd.DataFrame({"event": [1, 0, 2], "duration": [5, 10, 15]})
    y_pred = np.array(
        [
            [[0.2, 0.5], [0.8, 0.5]],
            [[0.7, 0.6], [0.3, 0.4]],
            # n_samples mismatch
        ]
    )
    time_grid = np.array([5, 10])

    with pytest.raises(ValueError, match="must have the same number of samples"):
        accuracy_in_time(y_test, y_pred, time_grid)


def test_non_increasing_time_grid():
    time_grid_0 = np.array([1, 3, 2])
    y_pred_0 = np.array([[[0.9, 0.4, 0.8], [0.1, 0.6, 0.2]]])

    time_grid = time_grid_0.copy()
    y_pred = y_pred_0.copy()

    y_test = pd.DataFrame(dict(event=[1], duration=[2]))
    with pytest.warns(UserWarning, match="time_grid is not sorted"):
        acc_in_time_1, taus_1 = accuracy_in_time(y_test, y_pred, time_grid)

    assert_array_equal(time_grid, time_grid_0)
    assert_array_equal(y_pred, y_pred_0)

    time_grid = np.array([1, 2, 3])
    y_pred = np.array([[[0.9, 0.8, 0.4], [0.1, 0.2, 0.6]]])
    acc_in_time_2, taus_2 = accuracy_in_time(y_test, y_pred, time_grid)
    assert_array_equal(acc_in_time_1, acc_in_time_2)
    assert_array_equal(taus_1, taus_2)
