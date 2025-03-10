import numpy as np
from scipy.interpolate import interp1d

from hazardous._km_sampler import _KaplanMeierSampler


def km_cal(y_conf, times, surv_prob_at_conf, return_diff_at_t=False):
    """
    Args:
        y (n_samples, 2): samples to fit the KM estimator
        times (array(n_times, )): array of times t at which to calculate the calibration
        surv_prob_at_conf (array(n_conf, n_times)): survival predictions at time t for
        D_{conf}

    Returns:
    """
    kaplan_sampler = _KaplanMeierSampler()
    kaplan_sampler.fit(y_conf)
    surv_func = kaplan_sampler.survival_func_

    times = np.sort(times)

    t_max = max(times)

    # global surv prob from KM
    surv_probs_KM = surv_func(times)
    # global surv prob from estimator
    surv_probs = surv_prob_at_conf.mean(axis=0)

    # Calculate calibration by integrating over times and
    # taking the difference between the survival probabilities
    # at time t and the survival probabilities at time t from KM
    diff_at_t = surv_probs - surv_probs_KM

    KM_cal = np.trapz(diff_at_t**2, times) / t_max
    if return_diff_at_t:
        return KM_cal, diff_at_t
    return KM_cal


def recalibrate_survival_function(
    X_conf,
    y_conf,
    times,
    estimator=None,
    X=None,
    surv_probs=None,
    surv_probs_conf=None,
    return_function=False,
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

    if estimator is None and (surv_probs is None or surv_probs_conf is None):
        raise ValueError(
            "Either estimator or (surv_probs and surv_probs_conf) must be provided"
        )

    # Calculate the survival probabilities to compute the calibration
    if surv_probs is None:
        if not hasattr(estimator, "predict_survival_function"):
            raise ValueError("Estimator must have a predict_survival_function method")

        surv_probs = estimator.predict_survival_function(X, times)
        surv_probs_conf = estimator.predict_survival_function(X_conf, times)

    # Calculate the calibration
    diff_at_t = km_cal(y_conf, times, surv_probs_conf, return_diff_at_t=True)[1]
    surv_probs_calibrated = surv_probs - diff_at_t

    if return_function:
        # Recalibrate the survival function
        return interp1d(
            x=times,
            y=surv_probs_calibrated,
            kind="previous",
            bounds_error=False,
            fill_value="extrapolate",
        )
    return surv_probs_calibrated
