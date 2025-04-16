import numpy as np

from hazardous._km_sampler import _AalenJohansenSampler
from hazardous.calibration._km_calibration import (
    km_calibration,
    recalibrate_survival_function,
    recalibrate_survival_function_predictions,
)
from hazardous.utils import check_y_survival


def aj_calibration(y, times, inc_prob_at_conf, return_diff_at_t=False):
    """
    Args:
        y (n_samples, 2): samples to fit the Aalen-Johansen estimator
        times (array(n_times, )): array of times at which to calculate the calibration
        inc_prob_at_conf (array(n_conf, n_events +1, n_times)): incidence
            predictions at `times` for D_{conf}

    Returns:
    """
    event, duration = check_y_survival(y)
    event_ids_ = np.array(sorted(list(set([0]) | set(event))))

    times = np.sort(times)

    aalen_sampler = _AalenJohansenSampler()
    aalen_sampler.fit(y)
    t_max = max(times)

    aj_calibrations = {}
    differences_at_t = {}

    km_cal, diff_at_t = km_calibration(
        y,
        times,
        surv_prob_at_conf=inc_prob_at_conf[:, 0, :],
        return_diff_at_t=True,
    )

    aj_calibrations[0] = km_cal
    differences_at_t[0] = diff_at_t

    for event_id in event_ids_[1:]:
        inc_func = aalen_sampler.incidence_func_[event_id]

        # global incidence probabilities from AJ
        inc_probs_AJ = inc_func(times)

        # global incidence probabilities from estimator
        incidence_probas_event = inc_prob_at_conf[:, event_id, :]
        inc_probs = incidence_probas_event.mean(axis=0)

        # Calculate calibration by integrating over times and
        # taking the difference between the survival probabilities
        # at time t and the survival probabilities at time t from KM
        diff_at_t = inc_probs - inc_probs_AJ
        aj_calibrations[event_id] = np.trapz(diff_at_t**2, times) / t_max
        differences_at_t[event_id] = diff_at_t
    if return_diff_at_t:
        return aj_calibrations, differences_at_t
    return aj_calibrations


def recalibrate_incidence_functions(
    estimator,
    X_conf,
    y_conf,
    X_train,
    return_function=False,
):
    """
    Args:
        X_conf (n_conf, n_features): samples to recalibrate the estimator
        y_conf (n_conf, 2): target
        estimator (BaseEstimator): trained estimator
        times (n_times): times to recalibrate the survival function

    Returns:
        estimator_calibrated:
    """
    event, duration = check_y_survival(y_conf)
    event_ids_ = np.array(sorted(list(set([0]) | set(event))))
    times = np.quantile(duration, np.linspace(0, 1, 100))

    inc_prob_conf = estimator.predict_cumulative_incidence(X_conf, times)
    inc_probs = estimator.predict_cumulative_incidence(X_train, times)
    # Calculate the survival probabilities to compute the calibration
    if not hasattr(estimator, "predict_cumulative_incidence"):
        raise ValueError("Estimator must have a predict_cumulative_incidence method")

    # Calculate the calibration
    differences_at_t = aj_calibration(
        y_conf, times, inc_prob_conf, return_diff_at_t=True
    )[1]

    recalibrated_inc_functions = []

    recalibrated_surv_probs = recalibrate_survival_function(
        X_conf,
        y_conf,
        times,
        estimator=estimator,
        X=X_train,
        surv_probs=inc_probs[:, 0, :],
        surv_probs_conf=inc_prob_conf[:, 0, :],
        return_function=False,
    )

    recalibrated_inc_functions.append(recalibrated_surv_probs)

    for event_id in event_ids_[1:]:
        diff_at_t = differences_at_t[event_id]
        inc_probs_calibrated = inc_probs[:, event_id, :] - diff_at_t[None, :]
        recalibrated_inc_functions.append(inc_probs_calibrated)

    return np.array(recalibrated_inc_functions).swapaxes(0, 1)


def recalibrate_incidence_functions_predictions(
    predictions_test,
    predictions_conf,
    times,
    y_conf,
    return_function=False,
):
    """
    Args:
        X_conf (n_conf, n_features): samples to recalibrate the estimator
        y_conf (n_conf, 2): target
        estimator (BaseEstimator): trained estimator
        times (n_times): times to recalibrate the survival function

    Returns:
        estimator_calibrated:
    """
    event, duration = check_y_survival(y_conf)
    event_ids_ = np.array(sorted(list(set([0]) | set(event))))

    # Calculate the calibration
    differences_at_t = aj_calibration(
        y_conf, times, predictions_conf, return_diff_at_t=True
    )[1]

    recalibrated_inc_functions = []

    recalibrated_surv_probs = recalibrate_survival_function_predictions(
        predictions_test[:, 0, :],
        predictions_conf[:, 0, :],
        y_conf,
        times,
        return_function=False,
    )

    recalibrated_inc_functions.append(recalibrated_surv_probs)

    for event_id in event_ids_[1:]:
        diff_at_t = differences_at_t[event_id]
        inc_probs_calibrated = predictions_test[:, event_id, :] - diff_at_t[None, :]
        recalibrated_inc_functions.append(inc_probs_calibrated)

    return np.array(recalibrated_inc_functions).swapaxes(0, 1)
