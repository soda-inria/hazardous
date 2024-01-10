import warnings

import numpy as np

from .._ipcw import IPCWEstimator, IPCWSampler
from ..utils import check_event_of_interest, check_y_survival


class IncidenceScoreComputer:
    """Censoring adjusted, time-dependent scoring rules.

    This class factorizes the computation of scoring rules such as the
    time-dependent Brier score for single-event or any event survival functions
    and cause-specific cumulative incidence functions.

    It leverages the Inverse Probability of Censoring Weighting (IPCW) scheme
    using a Kaplan-Meier of the censoring distribution to weight the terms.

    Parameters
    ----------
    y_train : np.array, dictionnary or dataframe
        The target, consisting in the 'event' and 'duration' columns. This is
        used to fit the IPCW estimator.

    event_of_interest : int or "any", default="any"
        The event to consider in a competing events setting.

        ``"any"`` indicates that all events except the censoring marker ``0``
        are considered collapsed together as a single event. In a single event
        setting, ``"any"`` and ``1`` are equivalent.

    """

    def __init__(
        self,
        y_train,
        event_of_interest="any",
        ipcw_est=None,
    ):
        self.y_train = y_train
        self.event_train, self.duration_train = check_y_survival(y_train)
        self.event_ids_ = np.unique(self.event_train)
        self.any_event_train = self.event_train > 0
        self.event_of_interest = event_of_interest

        y = dict(
            event=self.any_event_train,
            duration=self.duration_train,
        )
        # Estimate the censoring distribution from the training set
        # using Kaplan-Meier.
        if ipcw_est is None:
            self.ipcw_est = IPCWEstimator().fit(y)
        else:
            self.ipcw_est = ipcw_est.fit(y)

    def brier_score_survival(self, y_true, y_pred, times):
        """Time-dependent Brier score of a survival function estimate.

        Compute the time-dependent Brier score value for each individual and
        each time point in `times` and then average over individuals.

        This estimate is adjusted for censoring by leveraging the Inverse
        Probability of Censoring Weighting (IPCW) scheme.

        Parameters
        ----------
        y_true : record-array, dict or dataframe of shape (n_samples, 2)
            The ground truth, consisting in the 'event' and 'duration' columns.
            In a survival setting, we expect the event to be a binary
            indicator: 1 for the event of interest and 0 for censoring.
            Alternatively, all competing event types should be collapsed by
            setting event_of_interest="any".

        y_pred : array-like of shape (n_samples, n_times)
            Survival probability estimates predicted at ``times``. In the
            binary event settings, this is 1 - incidence_probability.

        times : array-like of shape (n_times)
            Times to estimate the survival probability and to compute the Brier
            Score.

        Returns
        -------
        brier_score : np.ndarray of shape (n_times)
            Time-dependent Brier scores averaged over the individuals.

        """
        if (self.event_ids_ > 0).sum() > 1 and self.event_of_interest != "any":
            warnings.warn(
                "Computing the survival Brier score only makes "
                "sense with a binary event indicator or when setting "
                "event_of_interest='any'. "
                "Instead this model is evaluated on data with event ids "
                f"{self.event_ids_.tolist()} and with "
                f"event_of_interest={self.event_of_interest}."
            )
        return self.brier_score_incidence(y_true, 1 - y_pred, times)

    def brier_score_incidence(self, y_true, y_pred, times):
        """Brier score for the cause-specific cumulative incidence function.

        Compute the Brier score values with IPCW adjustment for censoring for
        each cumulative incidence estimate for the event of interest and each
        requested time point and return the time-dependent Brier score averaged
        over individuals.

        Parameters
        ----------
        y_true : record-array, dictionnary or dataframe of shape (n_samples, 2)
            The ground truth, consisting in the 'event' and 'duration' columns.

        y_pred : array-like of shape (n_samples, n_times)
            Cause-specific cumulative incidence estimates predicted at
            ``times`` for the event of interest. In the single event type
            settings, or when event_of_interest == "any", this is 1 -
            survival_probability.

        times : array-like of shape (n_times)
            Times to estimate the survival probability and to compute the Brier
            score.

        Returns
        -------
        brier_score_incidence : np.ndarray
            Average value of the time-dependent Brier scores computed at time
            locations specified in the ``times`` argument.
        """
        event_true, duration_true = check_y_survival(y_true)
        check_event_of_interest(self.event_of_interest)

        if self.event_of_interest == "any":
            if y_true is self.y_train:
                event_true = self.any_event_train
            else:
                event_true = event_true > 0

        if y_pred.ndim != 2:
            raise ValueError(
                "'y_pred' must be a 2D array with shape (n_samples, n_times), got"
                f" shape {y_pred.shape}."
            )
        if y_pred.shape[0] != event_true.shape[0]:
            raise ValueError(
                "'y_true' and 'y_pred' must have the same number of samples, "
                f"got {event_true.shape[0]} and {y_pred.shape[0]} respectively."
            )
        if y_pred.shape[1] != times.shape[0]:
            raise ValueError(
                f"'times' length ({times.shape[0]}) "
                f"must be equal to y_pred.shape[1] ({y_pred.shape[1]})."
            )

        n_samples = event_true.shape[0]
        n_time_steps = times.shape[0]
        brier_scores = np.empty(
            shape=(n_samples, n_time_steps),
            dtype=np.float64,
        )
        ipcw_y = self.ipcw_est.compute_ipcw_at(duration_true)
        for t_idx, t in enumerate(times):
            y_true_binary, weights = self._weighted_binary_targets(
                y_event=event_true,
                y_duration=duration_true,
                times=np.full(shape=n_samples, fill_value=t),
                ipcw_y_duration=ipcw_y,
            )
            # XXX: refactor and rename this function to make it possible to
            # also compute the time-dependent binary cross-entropy loss.
            squared_error = (y_true_binary - y_pred[:, t_idx]) ** 2
            brier_scores[:, t_idx] = weights * squared_error

        return brier_scores.mean(axis=0)

    def integrated_brier_score_survival(self, y_true, y_pred, times):
        brier_scores = self.brier_score_survival(
            y_true,
            y_pred,
            times,
        )
        return self._time_integrated(brier_scores, times)

    def integrated_brier_score_incidence(self, y_true, y_pred, times):
        brier_scores = self.brier_score_incidence(
            y_true,
            y_pred,
            times,
        )
        return self._time_integrated(brier_scores, times)

    def _time_integrated(self, scores, times):
        ordering = np.argsort(times)
        sorted_times = times[ordering]
        sorted_scores = scores[ordering]
        time_span = sorted_times[-1] - sorted_times[0]
        return np.trapz(sorted_scores, sorted_times) / time_span

    def _weighted_binary_targets(self, y_event, y_duration, times, ipcw_y_duration):
        if self.event_of_interest == "any":
            # y should already be provided as binary indicator
            k = 1
        else:
            k = self.event_of_interest

        # Specify the binary classification target for each record in y and a
        # reference time horizon:
        #
        # - 1 when event of interest was observed before the reference time
        #   horizon,
        #
        # - 0 otherwise: any competing event happening at any time, censored
        #   record or event of interest happening after the reference time
        #   horizon.
        #
        #   Note: censored events only contribute (as negative target) when
        #   their duration is larger than the reference target horizon.
        #   Otherwise, they are discarded by setting their weight to 0 in the
        #   following.
        #
        #   Contrary to censored records, competing events always contribute as
        #   negative targets. There weight is always non-zero but differ if
        #   they happen either before or after the reference time horizon.
        #
        # This IPCW scheme for survival analysis (binary events) is described
        # in [Graf1999] and is extended to multiple competing events in
        # [Kretowska2018].
        event_k_before_horizon = (y_event == k) & (y_duration <= times)
        y_binary = event_k_before_horizon.astype(np.int32)

        ipcw_times = self.ipcw_est.compute_ipcw_at(times)
        any_event_or_censoring_after_horizon = y_duration > times
        weights = np.where(any_event_or_censoring_after_horizon, ipcw_times, 0)

        any_observed_event_before_horizon = (y_event > 0) & (y_duration <= times)
        weights = np.where(any_observed_event_before_horizon, ipcw_y_duration, weights)

        return y_binary, weights


def brier_score_survival(
    y_train,
    y_test,
    y_pred,
    times,
):
    r"""Time-dependent Brier score of a survival function estimate.

    .. math::

        \mathrm{BS}(t) = \frac{1}{n} \sum_{i=1}^n \mathbb{I}
        (y_i \leq t \land \delta_i = 1)
        \frac{(0 - \hat{S}(t | \mathbf{x}_i))^2}{\hat{G}(y_i)} +
        \mathbb{I}(y_i > t)
        \frac{(1 - \hat{S}(t | \mathbf{x}_i))^2}{\hat{G}(t)} ,

    where :math:`\hat{S}(t | \mathbf{x})` is the predicted probability of
    surviving up to time point :math:`t` for a feature vector :math:`\mathbf{x}`,
    and :math:`\hat{G}(t)` is the probability of remaining uncensored at time
    :math:`t`, estimated on the training set by the Kaplan-Meier estimator on the
    negation of the binary any-event indicator.

    Note that this assumes independence between censoring and the covariates.
    When this assumption is violated, the IPCW weights are biased and the Brier
    score is not a proper scoring rule anymore. See [Gerds2006]_ for a study of this
    bias.

    Parameters
    ----------
    y_train : record-array, dictionnary or dataframe of shape (n_samples, 2)
        The target, consisting in the 'event' and 'duration' columns. If the
        'event' column holds more than 1 event types, they are automatically
        collapsed to a single event type to compute the Brier score of the
        "any-event" survival function estimate.
        This is only used to estimate the IPCW values to adjust for censoring in
        the evaluation data.

    y_test : record-array, dictionnary or dataframe of shape (n_samples, 2)
        The ground truth, consisting in the 'event' and 'duration' columns.
        The same remark applies as for ``y_train`` with respect to the 'event'
        column.

    y_pred : array-like of shape (n_samples, n_times)
        Survival probability estimates predicted at ``times``.

    times : array-like of shape (n_times)
        Times at which the survival probability ``y_pred`` has been estimated
        and for which we compute the Brier score.

    See Also
    --------
    integrated_brier_score_survival : Time-integrated Brier score of a survival
        function estimate.

    Returns
    -------
    brier_score : np.ndarray of shape (n_times)

    References
    ----------
    .. [Graf1999] E. Graf, C. Schmoor, W. Sauerbrei, M. Schumacher, "Assessment
       and comparison of prognostic classification schemes for survival data",
       1999

    .. [Gerds2006] T. Gerds and M. Schumacher, "Consistent Estimation of the
       Expected Brier Score in General Survival Models with Right-Censored
       Event Times", 2006
    """
    computer = IncidenceScoreComputer(
        y_train,
        event_of_interest="any",
    )
    return computer.brier_score_survival(y_test, y_pred, times)


def integrated_brier_score_survival(
    y_train,
    y_test,
    y_pred,
    times,
):
    r"""Compute the Brier score integrated over the observed time range.

    .. math::

        \mathrm{IBS} = \frac{1}{t_{max} - t_{min}} \int^{t_{max}}_{t_{min}}
        \mathrm{BS}(u) du

    Note that this assumes independence between censoring and the covariates.
    When this assumption is violated, the IPCW weights are biased and the Brier
    score is not a proper scoring rule anymore. See [Gerds2006]_ for a study of
    this bias.

    Parameters
    ----------
    y_train : record-array, dictionnary or dataframe of shape (n_samples, 2)
        The target, consisting in the 'event' and 'duration' columns. If the
        'event' column holds more than 1 event types, they are automatically
        collapsed to a single event type to compute the Brier score of the
        "any-event" survival function estimate.
        This is only used to estimate the IPCW values to adjust for censoring in
        the evaluation data.

    y_test : record-array, dictionnary or dataframe of shape (n_samples, 2)
        The ground truth, consisting in the 'event' and 'duration' columns.
        The same remark applies as for ``y_train`` with respect to the 'event'
        column.

    y_pred : array-like of shape (n_samples, n_times)
        Survival probability estimates predicted at ``times``.

    times : array-like of shape (n_times)
        Times at which the survival probabilities ``y_pred`` has been estimated
        and for which we compute the Brier score.

    See Also
    --------
    brier_score_survival : Time-dependent Brier score of a survival function
        estimate.

    Returns
    -------
    ibs : float

    References
    ----------
    .. [Graf1999] E. Graf, C. Schmoor, W. Sauerbrei, M. Schumacher, "Assessment
       and comparison of prognostic classification schemes for survival data",
       1999

    .. [Gerds2006] T. Gerds and M. Schumacher, "Consistent Estimation of the
       Expected Brier Score in General Survival Models with Right-Censored
       Event Times", 2006
    """
    computer = IncidenceScoreComputer(
        y_train,
        event_of_interest="any",
    )
    return computer.integrated_brier_score_survival(y_test, y_pred, times)


def brier_score_incidence(
    y_train,
    y_test,
    y_pred,
    times,
    event_of_interest="any",
):
    r"""Time-dependent Brier score for the kth cause of event.

    .. math::

        \mathrm{BS}_k(t) = \frac{1}{n} \sum_{i=1}^n \hat{\omega}_i(t)
        (\mathbb{I}(t_i \leq t, \delta_i = k) - \hat{F}_k(t|\mathbf{x}_i))^2

    where :math:`\hat{F}_k(t | \mathbf{x}_i)` is an estimate of the
    (uncensored) cumulative incidence for the kth event up to time point
    :math:`t` for a feature vector :math:`\mathbf{x}_i` [Edwards2016]_:

    .. math::

            \hat{F}_k(t | \mathbf{x}_i) \approx P(T_i \leq t, E_i = k |
            \mathbf{x}_i)

    and :math:`\hat{\omega}_i(t)` are IPCW weigths based on the Kaplan-Meier
    estimate of the censoring distribution :math:`\hat{G}(t)`:

    .. math::

        \hat{\omega}_i(t)=\frac{\mathbb{I}(t_i \leq t, \delta_i \neq
        0)}{\hat{G}(t_i)} + \frac{\mathbb{I}(t_i > t)}{\hat{G}(t)}

    This scheme was introduced in [Graf1999]_ in the context of survival
    analysis and extended to competing events in [Kretowska2018]_.

    Note that this assumes independence between censoring and the covariates.
    When this assumption is violated, the IPCW weights are biased and the Brier
    score is not a proper scoring rule anymore. See [Gerds2006]_ for a study of
    this bias.

    Parameters
    ----------
    y_train : record-array, dictionnary or dataframe of shape (n_samples, 2)
        The target, consisting in the 'event' and 'duration' columns. This is
        used to fit the IPCW estimator.

    y_test : record-array, dictionnary or dataframe of shape (n_samples, 2)
        The ground truth, consisting in the 'event' and 'duration' columns. In
        the "event" column, `0` indicates censoring, and any other values
        indicate competing event types.

    y_pred : array-like of shape (n_samples, n_times)
        Incidence probability estimates predicted at ``times``. In the binary
        event settings, this is 1 - survival_probability.

    times : array-like of shape (n_times)
        Times at which the survival probability ``y_pred`` has been estimated
        and for which we compute the Brier score.

    event_of_interest : int or "any", default="any"
        The event to consider in a competing events setting. When an integer,
        this should be one of the non-zero values in the "event" column of
        ``y_train`` and ``y_test``.

        ``"any"`` indicates that all events except the censoring marker ``0``
        are considered collapsed together as a single event. In a single event
        setting, ``"any"`` and ``1`` are equivalent.

    Returns
    -------
    brier_score : np.ndarray of shape (n_times)

    See Also
    --------
    integrated_brier_score_incidence : Time-integrated Brier score for the kth
        cause of event.

    References
    ----------
    .. [Graf1999] E. Graf, C. Schmoor, W. Sauerbrei, M. Schumacher, "Assessment
       and comparison of prognostic classification schemes for survival data",
       1999

    .. [Kretowska2018] M. Kretowska, "Tree-based models for survival data with
       competing risks", 2018

    .. [Gerds2006] T. Gerds and M. Schumacher, "Consistent Estimation of the
       Expected Brier Score in General Survival Models with Right-Censored
       Event Times", 2006

    .. [Edwards2016] J. Edwards, L. Hester, M. Gokhale, C. Lesko,
       "Methodologic Issues When Estimating Risks in Pharmacoepidemiology.",
       2016, doi:10.1007/s40471-016-0089-1
    """
    # XXX: make times an optional kwarg to be compatible with
    # sksurv.metrics.brier_score?
    # XXX: 'times' must match the times of y_pred,
    # but we have no way to check that.
    # In this sense, 'y_pred[:, t_idx]' is incorrect when 'times'
    # is not the time used during the prediction.
    computer = IncidenceScoreComputer(
        y_train,
        event_of_interest=event_of_interest,
    )
    return computer.brier_score_incidence(y_test, y_pred, times)


def integrated_brier_score_incidence(
    y_train,
    y_test,
    y_pred,
    times,
    event_of_interest="any",
):
    r"""Time-integrated Brier score of a cause-specific cumulative incidence estimate.

    .. math::

        \mathrm{IBS}_k = \frac{1}{t_{max} - t_{min}} \int^{t_{max}}_{t_{min}}
        \mathrm{BS}_k(u) du

    This scheme was introduced in [Graf1999]_ for survival analysis and
    extended to competing events in [Kretowska2018]_.

    Note that this assumes independence between censoring and the covariates.
    When this assumption is violated, the IPCW weights are biased and the Brier
    score is not a proper scoring rule anymore. See [Gerds2006]_ for a study of
    this bias.

    Parameters
    ----------
    y_train : record-array, dictionnary or dataframe of shape (n_samples, 2)
        The target, consisting in the 'event' and 'duration' columns.
        This is used to fit the IPCW estimator.

    y_test : record-array, dictionnary or dataframe of shape (n_samples, 2)
        The ground truth, consisting in the 'event' and 'duration' columns.
        In the "event" column, `0` indicates censoring, and any other values
        indicate competing event types.

    y_pred : array-like of shape (n_samples, n_times)
        Incidence probability estimates predicted at ``times``.
        In the binary event settings, this is 1 - survival_probability.

    times : array-like of shape (n_times)
        Times at which the survival probabilities ``y_pred`` has been estimated
        and for which we compute the Brier score.

    event_of_interest : int or "any", default="any"
        The event to consider in a competing events setting. When an integer,
        this should be one of the non-zero values in the "event" column of
        ``y_train`` and ``y_test``.

        ``"any"`` indicates that all events except the censoring marker ``0``
        are considered collapsed together as a single event. In a single event
        setting, ``"any"`` and ``1`` are equivalent.

    Returns
    -------
    ibs : float

    See Also
    --------
    brier_score_incidence : Time-dependent Brier score for the kth cause of event.

    References
    ----------
    .. [Graf1999] E. Graf, C. Schmoor, W. Sauerbrei, M. Schumacher, "Assessment
       and comparison of prognostic classification schemes for survival data",
       1999

    .. [Kretowska2018] M. Kretowska, "Tree-based models for survival data with
       competing risks", 2018

    .. [Gerds2006] T. Gerds and M. Schumacher, "Consistent Estimation of the
       Expected Brier Score in General Survival Models with Right-Censored
       Event Times", 2006
    """
    computer = IncidenceScoreComputer(
        y_train,
        event_of_interest=event_of_interest,
    )
    return computer.integrated_brier_score_incidence(y_test, y_pred, times)


def brier_score_oracle_probas_incidence(
    y_train,
    y_test,
    y_pred,
    times,
    shape_censoring,
    scale_censoring,
    event_of_interest="any",
):
    ipcw_est = IPCWSampler(shape=shape_censoring, scale=scale_censoring)
    computer = IncidenceScoreComputer(
        y_train, event_of_interest=event_of_interest, ipcw_est=ipcw_est
    )
    return computer.brier_score_incidence(y_test, y_pred, times)


def integrated_brier_score_oracle_probas_incidence(
    y_train,
    y_test,
    y_pred,
    times,
    shape_censoring,
    scale_censoring,
    event_of_interest="any",
):
    ipcw_est = IPCWSampler(shape=shape_censoring, scale=scale_censoring)
    computer = IncidenceScoreComputer(
        y_train,
        event_of_interest=event_of_interest,
        ipcw_est=ipcw_est,
    )
    return computer.integrated_brier_score_incidence(y_test, y_pred, times)
