import numpy as np

from hazardous.utils import check_y_survival


def accuracy_in_time(y_test, y_pred, times, quantiles=None, taus=None):
    event_true, _ = check_y_survival(y_test)

    if y_pred.ndim != 3:
        raise ValueError(
            "'y_pred' must be a 3D array with shape (n_samples, n_events, n_times), got"
            f" shape {y_pred.shape}."
        )
    if y_pred.shape[0] != event_true.shape[0]:
        raise ValueError(
            "'y_true' and 'y_pred' must have the same number of samples, "
            f"got {event_true.shape[0]} and {y_pred.shape[0]} respectively."
        )
    times = np.atleast_1d(times)
    if y_pred.shape[1] != times.shape[0]:
        raise ValueError(
            f"'times' length ({times.shape[0]}) "
            f"must be equal to y_pred.shape[1] ({y_pred.shape[1]})."
        )

    if quantiles is not None:
        if taus is not None:
            raise ValueError("'quantiles' and 'taus' can't be set at the same time.")

        quantiles = np.atleast_1d(quantiles)
        if any(quantiles < 0) or any(quantiles > 1):
            raise ValueError(f"quantiles must be in [0, 1], got {quantiles}.")
        taus = np.quantile(times, quantiles)

    elif quantiles is None and taus is None:
        n_quantiles = min(times.shape[0], 8)
        quantiles = np.linspace(1 / n_quantiles, 1, n_quantiles)
        taus = np.quantile(times, quantiles)

    acc_in_time = []

    for tau in taus:
        mask_past_censored = (y_test["event"] == 0) & (y_test["duration"] < tau)

        tau_idx = np.searchsorted(times, tau)
        y_pred_at_t = y_pred[:, :, tau_idx]
        y_pred_class = y_pred_at_t[~mask_past_censored, :].argmax(axis=1)

        y_test_class = y_test["event"] * (y_test["duration"] < tau)
        y_test_class = y_test_class.loc[~mask_past_censored]

        acc_in_time.append((y_test_class.values == y_pred_class).mean())

    return acc_in_time, taus
