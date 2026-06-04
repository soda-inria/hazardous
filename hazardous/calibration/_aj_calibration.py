import numpy as np

from hazardous._km_sampler import _AalenJohansenSampler
from hazardous.calibration._km_calibration import km_calibration
from hazardous.utils import check_y_survival


def aj_calibration(y, times, inc_prob_at_conf, return_diff_at_t=False, alpha=2):
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
        aj_calibrations[event_id] = np.trapz(diff_at_t**alpha, times) / t_max
        differences_at_t[event_id] = diff_at_t
    if return_diff_at_t:
        return aj_calibrations, differences_at_t
    return aj_calibrations
