import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal

from hazardous.metrics import accuracy_in_time


def test_accuracy_in_time_basic():
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

    acc_in_time, taus = accuracy_in_time(y_test, y_pred, time_grid)
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


def test_quantiles_and_taus_set():
    """Test setting both quantiles and taus."""
    y_test = pd.DataFrame({"event": [1, 0, 2], "duration": [5, 10, 15]})
    y_pred = np.array(
        [
            [[0.2, 0.5], [0.8, 0.5]],
            [[0.7, 0.6], [0.3, 0.4]],
            [[0.1, 0.4], [0.9, 0.6]],
        ]
    )
    time_grid = np.array([5, 10])

    with pytest.raises(ValueError, match="'quantiles' and 'taus' can't be set"):
        accuracy_in_time(y_test, y_pred, time_grid, quantiles=[0.5], taus=[5])


def test_default_quantiles():
    """Test default quantile behavior."""
    y_test = pd.DataFrame({"event": [1, 0, 2], "duration": [5, 10, 15]})
    y_pred = np.array(
        [
            [[0.2, 0.5, 0.8], [0.8, 0.5, 0.2]],
            [[0.7, 0.6, 0.4], [0.3, 0.4, 0.6]],
            [[0.1, 0.4, 0.9], [0.9, 0.6, 0.1]],
        ]
    )
    time_grid = np.array([5, 10, 15])

    acc_in_time, taus = accuracy_in_time(y_test, y_pred, time_grid)

    assert len(taus) == 3  # Default quantiles should match the time grid length
    assert len(acc_in_time) == 3
