import numbers

import numpy as np
from scipy import stats
from sklearn.base import BaseEstimator, check_is_fitted
from sklearn.utils import check_array, check_consistent_length
from sklearn.utils._param_validation import Interval, StrOptions

from hazardous.utils import check_y_survival


def concordance_index_ipcw(
    y_train,
    y_test,
    y_pred_at_t,
    tau=None,
    tied_tol=1e-8,
):
    """Concordance index for right-censored data based on inverse probability \
        of censoring weights.

    This is an alternative to the estimator in :func:`concordance_index_censored`
    that does not depend on the distribution of censoring times in the test data.
    Therefore, the estimate is unbiased and consistent for a population concordance
    measure that is free of censoring.

    It is based on inverse probability of censoring weights, thus requires
    access to survival times from the training data to estimate the censoring
    distribution. Note that this requires that survival times `survival_test`
    lie within the range of survival times `survival_train`. This can be
    achieved by specifying the truncation time `tau`.
    The resulting `cindex` tells how well the given prediction model works in
    predicting events that occur in the time range from 0 to `tau`.

    The estimator uses the Kaplan-Meier estimator to estimate the
    censoring survivor function. Therefore, it is restricted to
    situations where the random censoring assumption holds and
    censoring is independent of the features.

    See the :ref:`User Guide </user_guide/evaluating-survival-models.ipynb>`
    and [1]_ for further description.

    Parameters
    ----------
    survival_train : structured array, shape = (n_train_samples,)
        Survival times for training data to estimate the censoring
        distribution from.
        A structured array containing the binary event indicator
        as first field, and time of event or time of censoring as
        second field.

    survival_test : structured array, shape = (n_samples,)
        Survival times of test data.
        A structured array containing the binary event indicator
        as first field, and time of event or time of censoring as
        second field.

    estimate : array-like, shape = (n_samples,)
        Estimated risk of experiencing an event of test data.

    tau : float, optional
        Truncation time. The survival function for the underlying
        censoring time distribution :math:`D` needs to be positive
        at `tau`, i.e., `tau` should be chosen such that the
        probability of being censored after time `tau` is non-zero:
        :math:`P(D > \\tau) > 0`. If `None`, no truncation is performed.

    tied_tol : float, optional, default: 1e-8
        The tolerance value for considering ties.
        If the absolute difference between risk scores is smaller
        or equal than `tied_tol`, risk scores are considered tied.

    Returns
    -------
    cindex : float
        Concordance index

    concordant : int
        Number of concordant pairs

    discordant : int
        Number of discordant pairs

    tied_risk : int
        Number of pairs having tied estimated risks

    tied_time : int
        Number of comparable pairs sharing the same time

    See also
    --------
    concordance_index_censored
        Simpler estimator of the concordance index.

    as_concordance_index_ipcw_scorer
        Wrapper class that uses :func:`concordance_index_ipcw`
        in its ``score`` method instead of the default
        :func:`concordance_index_censored`.

    References
    ----------
    .. [1] Uno, H., Cai, T., Pencina, M. J., D’Agostino, R. B., & Wei, L. J. (2011).
           "On the C-statistics for evaluating overall adequacy of risk prediction
           procedures with censored survival data".
           Statistics in Medicine, 30(10), 1105–1117.
    """
    test_event, test_time = check_y_survival(y_test)

    if tau is not None:
        mask = test_time < tau
        y_test = y_test.loc[mask]

    y_pred_at_t = _check_estimate_1d(y_pred_at_t, test_time)

    cens = CensoringDistributionEstimator()
    cens.fit(y_train)
    ipcw_test = cens.predict_ipcw(y_test)
    if tau is None:
        ipcw = ipcw_test
    else:
        ipcw = np.empty(y_pred_at_t.shape[0], dtype=ipcw_test.dtype)
        ipcw[mask] = ipcw_test
        ipcw[~mask] = 0

    w = np.square(ipcw)

    return _estimate_concordance_index(test_event, test_time, y_pred_at_t, w, tied_tol)


def _estimate_concordance_index(
    event_indicator, event_time, estimate, weights, tied_tol=1e-8
):
    order = np.argsort(event_time)

    tied_time = None

    concordant = 0
    discordant = 0
    tied_risk = 0
    numerator = 0.0
    denominator = 0.0
    for ind, mask, tied_time in _iter_comparable(event_indicator, event_time, order):
        est_i = estimate[order[ind]]
        event_i = event_indicator[order[ind]]
        w_i = weights[order[ind]]

        est = estimate[order[mask]]

        assert (
            event_i
        ), f"got censored sample at index {order[ind]}, but expected uncensored"

        ties = np.absolute(est - est_i) <= tied_tol
        n_ties = ties.sum()
        # an event should have a higher score
        con = est < est_i
        n_con = con[~ties].sum()

        numerator += w_i * n_con + 0.5 * w_i * n_ties
        denominator += w_i * mask.sum()

        tied_risk += n_ties
        concordant += n_con
        discordant += est.size - n_con - n_ties

    if tied_time is None:
        raise ValueError(
            "Data has no comparable pairs, cannot estimate concordance index."
        )

    cindex = numerator / denominator
    return cindex, concordant, discordant, tied_risk, tied_time


def _iter_comparable(event_indicator, event_time, order):
    n_samples = len(event_time)
    tied_time = 0
    i = 0
    while i < n_samples - 1:
        time_i = event_time[order[i]]
        end = i + 1
        while end < n_samples and event_time[order[end]] == time_i:
            end += 1

        # check for tied event times
        event_at_same_time = event_indicator[order[i:end]]
        censored_at_same_time = ~event_at_same_time
        for j in range(i, end):
            if event_indicator[order[j]]:
                mask = np.zeros(n_samples, dtype=bool)
                mask[end:] = True
                # an event is comparable to censored samples at same time point
                mask[i:end] = censored_at_same_time
                tied_time += censored_at_same_time.sum()
                yield (j, mask, tied_time)
        i = end


def _check_estimate_1d(estimate, test_time):
    estimate = check_array(estimate, ensure_2d=False, input_name="estimate")
    if estimate.ndim != 1:
        raise ValueError(
            f"Expected 1D array, got {estimate.ndim}D array"
            f" instead:\narray={estimate}.\n"
        )
    check_consistent_length(test_time, estimate)
    return estimate


def _compute_counts(event, time, order=None):
    """Count right censored and uncensored samples at each unique time point.

    Parameters
    ----------
    event : array
        Boolean event indicator.

    time : array
        Survival time or time of censoring.

    order : array or None
        Indices to order time in ascending order.
        If None, order will be computed.

    Returns
    -------
    times : array
        Unique time points.

    n_events : array
        Number of events at each time point.

    n_at_risk : array
        Number of samples that have not been censored or have not had an event
        at each time point.

    n_censored : array
        Number of censored samples at each time point.
    """
    n_samples = event.shape[0]

    if order is None:
        order = np.argsort(time, kind="mergesort")

    uniq_times = np.empty(n_samples, dtype=time.dtype)
    uniq_events = np.empty(n_samples, dtype=int)
    uniq_counts = np.empty(n_samples, dtype=int)

    i = 0
    prev_val = time[order[0]]
    j = 0
    while True:
        count_event = 0
        count = 0
        while i < n_samples and prev_val == time[order[i]]:
            if event[order[i]]:
                count_event += 1

            count += 1
            i += 1

        uniq_times[j] = prev_val
        uniq_events[j] = count_event
        uniq_counts[j] = count
        j += 1

        if i == n_samples:
            break

        prev_val = time[order[i]]

    times = np.resize(uniq_times, j)
    n_events = np.resize(uniq_events, j)
    total_count = np.resize(uniq_counts, j)
    n_censored = total_count - n_events

    # offset cumulative sum by one
    total_count = np.r_[0, total_count]
    n_at_risk = n_samples - np.cumsum(total_count)

    return times, n_events, n_at_risk[:-1], n_censored


def _compute_counts_truncated(event, time_enter, time_exit):
    """Compute counts for left truncated and right censored survival data.

    Parameters
    ----------
    event : array
        Boolean event indicator.

    time_start : array
        Time when a subject entered the study.

    time_exit : array
        Time when a subject left the study due to an
        event or censoring.

    Returns
    -------
    times : array
        Unique time points.

    n_events : array
        Number of events at each time point.

    n_at_risk : array
        Number of samples that are censored or have an event at each time point.
    """
    if (time_enter > time_exit).any():
        raise ValueError("exit time must be larger start time for all samples")

    n_samples = event.shape[0]

    uniq_times = np.sort(np.unique(np.r_[time_enter, time_exit]), kind="mergesort")
    total_counts = np.empty(len(uniq_times), dtype=int)
    event_counts = np.empty(len(uniq_times), dtype=int)

    order_enter = np.argsort(time_enter, kind="mergesort")
    order_exit = np.argsort(time_exit, kind="mergesort")
    s_time_enter = time_enter[order_enter]
    s_time_exit = time_exit[order_exit]

    t0 = uniq_times[0]
    # everything larger is included
    idx_enter = np.searchsorted(s_time_enter, t0, side="right")
    # everything smaller is excluded
    idx_exit = np.searchsorted(s_time_exit, t0, side="left")

    total_counts[0] = idx_enter
    # except people die on the day they enter
    event_counts[0] = 0

    for i in range(1, len(uniq_times)):
        ti = uniq_times[i]

        while idx_enter < n_samples and s_time_enter[idx_enter] < ti:
            idx_enter += 1

        while idx_exit < n_samples and s_time_exit[idx_exit] < ti:
            idx_exit += 1

        risk_set = np.setdiff1d(
            order_enter[:idx_enter], order_exit[:idx_exit], assume_unique=True
        )
        total_counts[i] = len(risk_set)

        count_event = 0
        k = idx_exit
        while k < n_samples and s_time_exit[k] == ti:
            if event[order_exit[k]]:
                count_event += 1
            k += 1
        event_counts[i] = count_event

    return uniq_times, event_counts, total_counts


def _ci_logmlog(prob_survival, sigma_t, z):
    """Compute the pointwise log-minus-log transformed confidence intervals"""
    eps = np.finfo(prob_survival.dtype).eps
    log_p = np.zeros_like(prob_survival)
    np.log(prob_survival, where=prob_survival > eps, out=log_p)
    theta = np.zeros_like(prob_survival)
    np.true_divide(sigma_t, log_p, where=log_p < -eps, out=theta)
    theta = np.array([[-1], [1]]) * theta * z
    ci = np.exp(np.exp(theta) * log_p)
    ci[:, prob_survival <= eps] = 0.0
    ci[:, 1.0 - prob_survival <= eps] = 1.0
    return ci


def _km_ci_estimator(prob_survival, ratio_var, conf_level, conf_type):
    if conf_type not in {"log-log"}:
        raise ValueError(
            "conf_type must be None or a str among {'log-log'}, but was"
            f" {conf_type!r}"
        )

    if (
        not isinstance(conf_level, numbers.Real)
        or not np.isfinite(conf_level)
        or conf_level <= 0
        or conf_level >= 1.0
    ):
        raise ValueError(
            "conf_level must be a float in the range (0.0, 1.0), but was"
            f" {conf_level!r}"
        )

    z = stats.norm.isf((1.0 - conf_level) / 2.0)
    sigma = np.sqrt(np.cumsum(ratio_var))
    ci = _ci_logmlog(prob_survival, sigma, z)
    return ci


def kaplan_meier_estimator(
    event,
    time_exit,
    time_enter=None,
    time_min=None,
    reverse=False,
    conf_level=0.95,
    conf_type=None,
):
    """Kaplan-Meier estimator of survival function.

    See [1]_ for further description.

    Parameters
    ----------
    event : array-like, shape = (n_samples,)
        Contains binary event indicators.

    time_exit : array-like, shape = (n_samples,)
        Contains event/censoring times.

    time_enter : array-like, shape = (n_samples,), optional
        Contains time when each individual entered the study for
        left truncated survival data.

    time_min : float, optional
        Compute estimator conditional on survival at least up to
        the specified time.

    reverse : bool, optional, default: False
        Whether to estimate the censoring distribution.
        When there are ties between times at which events are observed,
        then events come first and are subtracted from the denominator.
        Only available for right-censored data, i.e. `time_enter` must
        be None.

    conf_level : float, optional, default: 0.95
        The level for a two-sided confidence interval on the survival curves.

    conf_type : None or {'log-log'}, optional, default: 'log-log'.
        The type of confidence intervals to estimate.
        If `None`, no confidence intervals are estimated.
        If "log-log", estimate confidence intervals using
        the log hazard or :math:`log(-log(S(t)))` as described in [2]_.

    Returns
    -------
    time : array, shape = (n_times,)
        Unique times.

    prob_survival : array, shape = (n_times,)
        Survival probability at each unique time point.
        If `time_enter` is provided, estimates are conditional probabilities.

    conf_int : array, shape = (2, n_times)
        Pointwise confidence interval of the Kaplan-Meier estimator
        at each unique time point.
        Only provided if `conf_type` is not None.

    Examples
    --------
    Creating a Kaplan-Meier curve:

    >>> x, y, conf_int = kaplan_meier_estimator(event, time, conf_type="log-log")
    >>> plt.step(x, y, where="post")
    >>> plt.fill_between(x, conf_int[0], conf_int[1], alpha=0.25, step="post")
    >>> plt.ylim(0, 1)
    >>> plt.show()

    See also
    --------
    sksurv.nonparametric.SurvivalFunctionEstimator
        Estimator API of the Kaplan-Meier estimator.

    References
    ----------
    .. [1] Kaplan, E. L. and Meier, P., "Nonparametric estimation from
           incomplete observations", Journal of The American Statistical Association,
           vol. 53, pp. 457-481, 1958.
    .. [2] Borgan Ø. and Liestøl K., "A Note on Confidence Intervals and Bands for the
           Survival Function Based on Transformations", Scandinavian Journal of
           Statistics. 1990;17(1):35–41.
    """
    check_consistent_length(event, time_enter, time_exit)

    if conf_type is not None and reverse:
        raise NotImplementedError(
            "Confidence intervals of the censoring distribution is not implemented."
        )

    if time_enter is None:
        uniq_times, n_events, n_at_risk, n_censored = _compute_counts(event, time_exit)

        if reverse:
            n_at_risk -= n_events
            n_events = n_censored
    else:
        if reverse:
            raise ValueError(
                "The censoring distribution cannot be estimated from left truncated"
                " data"
            )

        uniq_times, n_events, n_at_risk = _compute_counts_truncated(
            event, time_enter, time_exit
        )

    # account for 0/0 = nan
    ratio = np.divide(
        n_events,
        n_at_risk,
        out=np.zeros(uniq_times.shape[0], dtype=float),
        where=n_events != 0,
    )
    values = 1.0 - ratio

    if conf_type is not None:
        ratio_var = np.divide(
            n_events,
            n_at_risk * (n_at_risk - n_events),
            out=np.zeros(uniq_times.shape[0], dtype=float),
            where=(n_events != 0) & (n_at_risk != n_events),
        )

    if time_min is not None:
        mask = uniq_times >= time_min
        uniq_times = np.compress(mask, uniq_times)
        values = np.compress(mask, values)

    prob_survival = np.cumprod(values)

    if conf_type is None:
        return uniq_times, prob_survival

    if time_min is not None:
        ratio_var = np.compress(mask, ratio_var)

    ci = _km_ci_estimator(prob_survival, ratio_var, conf_level, conf_type)

    return uniq_times, prob_survival, ci


def nelson_aalen_estimator(event, time):
    """Nelson-Aalen estimator of cumulative hazard function.

    See [1]_, [2]_ for further description.

    Parameters
    ----------
    event : array-like, shape = (n_samples,)
        Contains binary event indicators.

    time : array-like, shape = (n_samples,)
        Contains event/censoring times.

    Returns
    -------
    time : array, shape = (n_times,)
        Unique times.

    cum_hazard : array, shape = (n_times,)
        Cumulative hazard at each unique time point.

    References
    ----------
    .. [1] Nelson, W., "Theory and applications of hazard plotting for
           censored failure data", Technometrics, vol. 14, pp. 945-965, 1972.

    .. [2] Aalen, O. O., "Nonparametric inference for a family of counting processes",
           Annals of Statistics, vol. 6, pp. 701–726, 1978.
    """
    event, time = check_y_survival(event, time)
    check_consistent_length(event, time)
    uniq_times, n_events, n_at_risk, _ = _compute_counts(event, time)

    y = np.cumsum(n_events / n_at_risk)

    return uniq_times, y


def ipc_weights(event, time):
    """Compute inverse probability of censoring weights

    Parameters
    ----------
    event : array, shape = (n_samples,)
        Boolean event indicator.

    time : array, shape = (n_samples,)
        Time when a subject experienced an event or was censored.

    Returns
    -------
    weights : array, shape = (n_samples,)
        inverse probability of censoring weights

    See also
    --------
    CensoringDistributionEstimator
        An estimator interface for estimating inverse probability
        of censoring weights for unseen time points.
    """
    if event.all():
        return np.ones(time.shape[0])

    unique_time, p = kaplan_meier_estimator(event, time, reverse=True)

    idx = np.searchsorted(unique_time, time[event])
    Ghat = p[idx]

    assert (Ghat > 0).all()

    weights = np.zeros(time.shape[0])
    weights[event] = 1.0 / Ghat

    return weights


class SurvivalFunctionEstimator(BaseEstimator):
    """Kaplan–Meier estimate of the survival function.

    Parameters
    ----------
    conf_level : float, optional, default: 0.95
        The level for a two-sided confidence interval on the survival curves.

    conf_type : None or {'log-log'}, optional, default: 'log-log'.
        The type of confidence intervals to estimate.
        If `None`, no confidence intervals are estimated.
        If "log-log", estimate confidence intervals using
        the log hazard or :math:`log(-log(S(t)))`.

    See also
    --------
    sksurv.nonparametric.kaplan_meier_estimator
        Functional API of the Kaplan-Meier estimator.
    """

    _parameter_constraints = {
        "conf_level": [Interval(numbers.Real, 0.0, 1.0, closed="neither")],
        "conf_type": [None, StrOptions({"log-log"})],
    }

    def __init__(self, conf_level=0.95, conf_type=None):
        self.conf_level = conf_level
        self.conf_type = conf_type

    def fit(self, y):
        """Estimate survival distribution from training data.

        Parameters
        ----------
        y : structured array, shape = (n_samples,)
            A structured array containing the binary event indicator
            as first field, and time of event or time of censoring as
            second field.

        Returns
        -------
        self
        """
        self._validate_params()
        event, time = check_y_survival(y)

        values = kaplan_meier_estimator(
            event, time, conf_level=self.conf_level, conf_type=self.conf_type
        )
        if self.conf_type is None:
            unique_time, prob = values
        else:
            unique_time, prob, conf_int = values
            self.conf_int_ = np.column_stack((np.ones((2, 1)), conf_int))

        self.unique_time_ = np.r_[-np.infty, unique_time]
        self.prob_ = np.r_[1.0, prob]

        return self

    def predict_proba(self, time, return_conf_int=False):
        """Return probability of an event after given time point.

        :math:`\\hat{S}(t) = P(T > t)`

        Parameters
        ----------
        time : array, shape = (n_samples,)
            Time to estimate probability at.

        return_conf_int : bool, optional, default: False
            Whether to return the pointwise confidence interval
            of the survival function.
            Only available if :meth:`fit()` has been called
            with the `conf_type` parameter set.

        Returns
        -------
        prob : array, shape = (n_samples,)
            Probability of an event at the passed time points.

        conf_int : array, shape = (2, n_samples)
            Pointwise confidence interval at the passed time points.
            Only provided if `return_conf_int` is True.
        """
        check_is_fitted(self, "unique_time_")
        if return_conf_int and not hasattr(self, "conf_int_"):
            raise ValueError(
                "If return_conf_int is True, SurvivalFunctionEstimator must be fitted"
                " with conf_int != None"
            )

        time = check_array(time, ensure_2d=False, estimator=self, input_name="time")

        # K-M is undefined if estimate at last time point is non-zero
        extends = time > self.unique_time_[-1]
        if self.prob_[-1] > 0 and extends.any():
            raise ValueError(
                "time must be smaller than largest observed time point:"
                f" {self.unique_time_[-1]}"
            )

        # beyond last time point is zero probability
        Shat = np.empty(time.shape, dtype=float)
        Shat[extends] = 0.0

        valid = ~extends
        time = time[valid]
        idx = np.searchsorted(self.unique_time_, time)
        # for non-exact matches, we need to shift the index to left
        eps = np.finfo(self.unique_time_.dtype).eps
        exact = np.absolute(self.unique_time_[idx] - time) < eps
        idx[~exact] -= 1
        Shat[valid] = self.prob_[idx]

        if not return_conf_int:
            return Shat

        ci = np.empty((2, time.shape[0]), dtype=float)
        ci[:, extends] = np.nan
        ci[:, valid] = self.conf_int_[:, idx]
        return Shat, ci


class CensoringDistributionEstimator(SurvivalFunctionEstimator):
    """Kaplan–Meier estimator for the censoring distribution."""

    def fit(self, y):
        """Estimate censoring distribution from training data.

        Parameters
        ----------
        y : structured array, shape = (n_samples,)
            A structured array containing the binary event indicator
            as first field, and time of event or time of censoring as
            second field.

        Returns
        -------
        self
        """
        event, time = check_y_survival(y)
        if event.all():
            self.unique_time_ = np.unique(time)
            self.prob_ = np.ones(self.unique_time_.shape[0])
        else:
            unique_time, prob = kaplan_meier_estimator(event, time, reverse=True)
            self.unique_time_ = np.r_[-np.infty, unique_time]
            self.prob_ = np.r_[1.0, prob]

        return self

    def predict_ipcw(self, y):
        """Return inverse probability of censoring weights at given time points.

        :math:`\\omega_i = \\delta_i / \\hat{G}(y_i)`

        Parameters
        ----------
        y : structured array, shape = (n_samples,)
            A structured array containing the binary event indicator
            as first field, and time of event or time of censoring as
            second field.

        Returns
        -------
        ipcw : array, shape = (n_samples,)
            Inverse probability of censoring weights.
        """
        event, time = check_y_survival(y)
        Ghat = self.predict_proba(time[event])

        if (Ghat == 0.0).any():
            raise ValueError(
                "censoring survival function is zero at one or more time points"
            )

        weights = np.zeros(time.shape[0])
        weights[event] = 1.0 / Ghat

        return weights
