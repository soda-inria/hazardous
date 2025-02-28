import numpy as np

from hazardous._km_sampler import _AalenJohansenSampler
from hazardous.calibration._km_calibration import km_cal, recalibrate_survival_function
from hazardous.utils import check_y_survival


def aj_cal(y, times, inc_prob_at_conf, return_diff_at_t=False):
    """
    Args:
        y (n_samples, 2): samples to fit the Aalen-Johansen estimator
        times (array(n_times, )): array of times t at which to calculate the calibration
            inc_prob_at_conf (array(n_conf, n_events +1, n_times)): incidence
            predictions at time t for D_{conf}

    Returns:
    """
    event, duration = check_y_survival(y)
    event_ids_ = np.array(sorted(list(set([0]) | set(event))))

    aalen_sampler = _AalenJohansenSampler()
    aalen_sampler.fit(y)
    t_max = max(times)

    AJ_calibrations = {}
    differences_at_t = {}

    KM_cal, diff_at_t = km_cal(
        y,
        times,
        surv_prob_at_conf=inc_prob_at_conf[0],
        return_diff_at_t=True,
    )

    AJ_calibrations[0] = KM_cal
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
        AJ_calibrations[event_id] = np.trapz(diff_at_t**2, times) / t_max
        differences_at_t[event_id] = diff_at_t
    if return_diff_at_t:
        return AJ_calibrations, differences_at_t
    return AJ_calibrations


def recalibrate_incidence_functions(
    X, y, X_conf, times, estimator=None, inc_probs=None, inc_prob_at_conf=None
):
    """
    Args:
        X (n_conf, n_features): samples to recalibrate the estimator
        y (n_conf, 2): target
        estimator (BaseEstimator): trained estimator
        times (n_times): times to recalibrate the survival function

    Returns:
        estimator_calibrated:
    """
    event, duration = check_y_survival(y)
    event_ids_ = np.array(sorted(list(set([0]) | set(event))))

    if estimator is None and (inc_probs is None or inc_prob_at_conf is None):
        raise ValueError(
            "Either estimator or (inc_prob and inc_prob_at_conf) must be provided"
        )

    # Calculate the survival probabilities to compute the calibration
    if inc_probs is None:
        if not hasattr(estimator, "predict_cumulative_incidence"):
            raise ValueError(
                "Estimator must have a predict_cumulative_incidence method"
            )

        inc_probs = estimator.predict_cumulative_incidence(X, times)
    if inc_prob_at_conf is None:
        if not hasattr(estimator, "predict_cumulative_incidence"):
            raise ValueError(
                "Estimator must have a predict_cumulative_incidence method"
            )
        inc_prob_at_conf = estimator.predict_cumulative_incidence(X_conf, times)

    # Calculate the calibration
    differences_at_t = aj_cal(y, times, inc_prob_at_conf, return_diff_at_t=True)[1]
    recalibrated_inc_functions = []

    recalibrated_surv_probs = recalibrate_survival_function(
        X,
        y,
        X_conf,
        times,
        surv_probs=inc_probs[:, 0, :],
        surv_probs_conf=inc_prob_at_conf[:, 0, :],
        return_function=False,
    )

    recalibrated_inc_functions.append(recalibrated_surv_probs)

    for event_id in event_ids_[1:]:
        diff_at_t = differences_at_t[event_id]
        inc_probs_calibrated = inc_probs[:, event_id, :] - diff_at_t

        # Recalibrate the survival function
        recalibrated_inc_functions.append(inc_probs_calibrated)

    return np.array(recalibrated_inc_functions).swapaxes(0, 1)
