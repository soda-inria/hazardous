import warnings
from collections import Counter, defaultdict

import numpy as np
from scipy.interpolate import interp1d

from .._ipcw import KaplanMeierIPCW
from ..utils import check_y_survival


def concordance_index_incidence(
    y_test,
    y_pred,
    y_train=None,
    ipcw_estimator="km",
    time_grid=None,
    taus=None,
    event_of_interest=1,
    tied_tol=1e-8,
):
    r"""Time-dependent concordance index for prognostic models with competing risks \
    using inverse probability of censoring weighting.

    .. math::

        \mathrm{C}(t) = \frac{\sum_{i=1}^n \sum_{j=1}^n (\tilde{A}_{ij}
        \hat{W}_{ij, 1}^{-1} + \tilde{B}_{ij} \hat{W}_{ij, 2}^{-1}) Q^{ij}(t)
        \tilde{N}^1_i(t)}
        {\sum_{i=1}^n \sum_{j=1}^n (\tilde{A}_{ij}
        \hat{W}_{ij, 1}^{-1} + \tilde{B}_{ij} \hat{W}_{ij, 2}^{-1}) \tilde{N}^1_i(t)}

    where:

    .. math::

        \begin{align}
        \tilde{N}^1_i(t) &= I\{\tilde{T}_i \leq t, \tilde{D}_i = 1\} \\
        \tilde{A}_{ij} &= I\{\tilde{T}_i < \tilde{T}_j \cup (\tilde{T}_i =
        \tilde{T}_j \cap D_j = 0)\} \\
        \tilde{B}_{ij} &= I\{\tilde{T}_i \geq \tilde{T}_j, D_j = 2\} \\
        \hat{W}_{ij,1} &= \hat{G}(\tilde{T}_i-|X_i) \hat{G}(\tilde{T}_i|X_j) \\
        \hat{W}_{ij,2} &= \hat{G}(\tilde{T}_i-|X_i) \hat{G}(\tilde{T}_j-|X_j) \\
        Q_{ij}(t) &= I\{M(t, X_i) > M(t, X_j)\}
        \end{align}

    where :math:`D_j = 0`, :math:`D_j = 1` and :math:`D_j = 2` respectively indicate
    individuals having been censored, individuals having experienced the event of
    interest, and individual having experienced a competing event. :math:`\hat{G}`
    is a IPCW estimator, :math:`Q_{ij}(t)` is an indicator for the order of
    predicted risk at :math:`t`, and :math:`M` is the predicted cumulative incidence
    function for the event of interest.

    The concordance index (C-index) is a common metric in survival analysis that
    evaluates whether the model predictions correctly order pairs of individuals with
    respect to the timing of the event of interest. It is defined as the probability
    that the prediction is concordant for a pair of individuals and is computed as the
    ratio of the number of concordant pairs to the total number of pairs.
    This implementation extends the C-index to the competing risks setting, where
    multiple alternative events are considered, aiming to determine which event occurs
    first and when, following the formulas and notations in [Wolbers2014]_.

    Due to the right-censoring in the data, the order of some pairs is unknown,
    so we define the notion of comparable pairs, i.e. the pairs for which
    we can compare the order of occurrence of the event of interest.
    A pair :math:`(i, j)` is comparable, with :math:`i` experiencing the event of
    interest at time :math:`T_i` if:

    - :math:`j` is censored or experience any event at a strictly greater time
      :math:`T_j > T_i` (pair of type A)
    - :math:`j` is censored at the exact same time :math:`T_i = T_j` (pair of type A).
    - :math:`j` experiences a competing event before or at time :math:`T_i`
      (pair of type B)

    A pair is then considered concordant if the predicted incidence of the event of
    interest for :math:`i` at time :math:`\tau` is larger than the predicted incidence
    for :math:`j`. If both predicted incidences are ties, the pair is counted as
    :math:`1/2` for the count of concordant pairs.

    The C-index has been shown to depend on the censoring distribution, and an
    inverse probability of censoring weighting (IPCW) allows to overcome this
    limitation [Uno2011]_, [Gerds2013]_. By default, the IPCW is implemented with a
    Kaplan-Meier estimator. Additionnally, the C-index is not a proper metric to
    evaluate a survival model [Blanche2019]_, and alternatives as the integrated Brier
    score (``integrated_brier_score_incidence``) should be considered.
    The C-index is a ranking metric and, unlike proper scoring rule metrics,
    cannot evaluate the calibration of the estimator.

    Parameters
    ----------
    y_test : array, dictionnary or dataframe of shape (n_samples, 2)
        The test target, consisting in the 'event' and 'duration' columns.

    y_pred: array of shape (n_samples, n_time_grid)
        Cumulative incidence for the event of interest, at the time points
        from the input time_grid.

    y_train : array, dictionnary or dataframe of shape (n_samples, 2), default=None
        The train target, consisting in the 'event' and 'duration' columns.
        Only used when ipcw_estimator = "km".

    ipcw_estimator : None, 'km', or fitted estimator, default="km"
        The inverse probability of censoring weighted (IPCW) estimator.

        - Pass None to set uniform weights to all samples.
        - Pass "km" to use the Kaplan-Meier IPCW estimator. It fits using y_train,
          which must be set.

    time_grid: array of shape (n_time_grid,), default=None
        Time points used to predict the cumulative incidence.

    taus: float or array of shape (n_taus,), default=None
        Timepoint(s) at which the concordance index is evaluated.

    event_of_interest: int, default=1
        For competing risks, the event of interest.

    tied_tol : float, default=1e-8
        The tolerance range to consider two probabilities equal.

    Returns
    -------
    cindex: array of shape (n_taus,)
        Value of the concordance index for each tau in taus.

    References
    ----------
    .. [Wolbers2014] M. Wolbers, P. Blanche, M. T. Koller, J. C. Witteman, T. A. Gerds,
       "Concordance for prognostic models with competing risks", 2014

    .. [Uno2011] H. Uno, T. Cai, M. J. Pencina, R. B. D'Agostino,  L. J. Wei, "On the
       C-statistics for evaluating overall adequacy of risk prediction
       procedures with censored survival data", 2011

    .. [Gerds2013] T. A. Gerds, M. W. Kattan, M. Schumacher, C. Yu, "Estimating a
       time-dependent concordance index for survival prediction models
       with covariate dependent censoring", 2013

    .. [Blanche2019] P. Blanche, M. W. Kattan, T. A. Gerds, "The c-index is not proper
       for the evaluation of-year predicted risks", 2019
    """
    c_index_report = _concordance_index_incidence_report(
        y_test,
        y_pred,
        y_train=y_train,
        ipcw_estimator=ipcw_estimator,
        time_grid=time_grid,
        taus=taus,
        event_of_interest=event_of_interest,
        tied_tol=tied_tol,
    )
    return np.array(c_index_report["cindex"])


def _concordance_index_incidence_report(
    y_test,
    y_pred,
    y_train=None,
    ipcw_estimator="km",
    time_grid=None,
    taus=None,
    event_of_interest=1,
    tied_tol=1e-8,
):
    """Report version of function ``concordance_index_incidence``.

    Running this function directly is useful to get more insights about
    the underlying statistics of the C-index.
    All returned lists have the same length as taus.

    Returns
    -------
    cindex: list of float
        Value of the concordance index

    n_pairs_a: list of int
        Number of comparable pairs with (T_i < T_j) | ((T_i = T_j) & (D_j = 0))
        (type A).
        Those are the only comparable pairs for survival without competing events.

    n_concordant_pairs_a: list of int
        Number of concordant pairs among A pairs without ties.

    n_ties_times: list of int
        Number of pairs which experienced the event of interest at the same time
        (D_i = D_j = 1) & (T_i = T_j).

    n_ties_pred_a: list of int
        Number of pairs of type A with np.abs(y_pred_i - y_pred_j) <= tied_tol

    n_pairs_b: list of int
        Number of comparable pairs with T_i >= T_j where j has
        a competing event (type B).
        0 if there are no competing events.

    n_concordant_pairs_b: list of int
        Number of concordant pairs among B pairs without ties.

    n_ties_pred_b: list of int
        Number of pairs of type B with np.abs(y_pred_i - y_pred_j) <= tied_tol.
    """
    if y_pred.ndim != 2:
        raise ValueError(
            "y_pred dimension must be 2, with shape (n_samples, n_time_grid), "
            f"got {y_pred.ndim=}."
        )
    if y_test.shape[0] != y_pred.shape[0]:
        raise ValueError(
            "y_test and y_pred must be the same length, got:"
            f"{y_test.shape[0]=} and {y_pred.shape[0]=}."
        )
    if time_grid is not None and len(time_grid) != y_pred.shape[1]:
        raise ValueError(
            "y_pred must have the same number of columns as the length of time_grid, "
            f"got: {y_test.shape[1]=} and {len(time_grid)=}."
        )

    if ipcw_estimator is None:
        if y_train is not None:
            warnings.warn(
                "y_train passed but ipcw_estimator is set to None, "
                "therefore y_train won't be used. Set y_train=None "
                "to silence this warning."
            )
        ipcw = np.ones(y_test["event"].shape[0])

    else:
        if y_train is None:
            # Raising here since the error raised by the IPCW estimator doesn't
            # help the user understand that y_train is missing.
            raise ValueError(
                "ipcw_estimator is set, but y_train is None. "
                "Set y_train to fix this error."
            )
        # TODO: add cox option
        ipcw_estimator_ = KaplanMeierIPCW().fit(y_train)
        ipcw = ipcw_estimator_.compute_ipcw_at(
            y_test["duration"]
        )  # shape: (n_samples_test,)

    if taus is None:
        t_min, t_max = 0, y_test["duration"].max()
        taus = [t_max]
        if time_grid is None:
            time_grid = np.linspace(t_min, t_max, num=y_pred.shape[1])
    else:
        if time_grid is None:
            raise ValueError("When 'taus' is set, 'time_grid' must also be set.")
    taus = np.atleast_1d(taus)

    c_index_report = defaultdict(list)
    c_index_report["taus"] = taus

    for tau in taus:
        y_pred_tau = interpolate_preds(y_pred, time_grid, tau)
        y_test_tau = y_test.copy()
        y_test_tau.loc[y_test_tau["duration"] > tau, "event"] = 0
        c_index_report_tau = _concordance_index_tau(
            y_test_tau, y_pred_tau, ipcw, event_of_interest, tied_tol
        )
        for metric_name, metric_value in c_index_report_tau.items():
            c_index_report[metric_name].append(metric_value)

    return c_index_report


def _concordance_index_tau(y_test, y_pred, ipcw, event_of_interest, tied_tol=1e-8):
    """Compute the C-index and the associated statistics for a given tau."""
    event, duration = check_y_survival(y_test)
    if not (event == event_of_interest).any():
        warnings.warn(
            f"There is not any event for {event_of_interest=!r}. "
            "The C-index is undefined."
        )

    stats_a = _StatsComputerTypeA().compute(
        event=event,
        duration=duration,
        y_pred=y_pred,
        ipcw=ipcw,
        event_of_interest=event_of_interest,
        tied_tol=tied_tol,
    )

    is_competing_event = (~np.isin(event, [0, event_of_interest])).any()
    if is_competing_event:
        stats_b = _StatsComputerTypeB().compute(
            event=event,
            duration=duration,
            y_pred=y_pred,
            ipcw=ipcw,
            event_of_interest=event_of_interest,
            tied_tol=tied_tol,
        )
    else:
        stats_b = Counter()

    stats = stats_a + stats_b

    if stats["weighted_pairs"] == 0:
        stats["weighted_pairs"] = np.nan

    cindex = (
        stats["weighted_concordant_pairs"] + 0.5 * stats["weighted_ties_pred"]
    ) / stats["weighted_pairs"]

    keys = ["n_pairs", "n_concordant_pairs", "n_ties_pred"]

    return {
        "cindex": cindex,
        "n_ties_times": stats_a["n_ties_times"],
        **{f"{k}_a": stats_a[k] for k in keys},
        **{f"{k}_b": stats_b[k] for k in keys},
    }


class _StatsComputer:
    def compute(
        self,
        event,
        duration,
        y_pred,
        ipcw,
        event_of_interest,
        tied_tol=1e-8,
    ):
        """Compute the C-index with a quadratic time complexity."""
        # note that event is obtained from y_test_tau thus event[i] = 0 if T_i > tau
        event = self._preprocess_event(event, event_of_interest)
        # now event[i] is equal to 
        # 0 if T_i > tau
        # 0 if T_i <= tau and D_i = censoring
        # 1 if T_i <= tau and D_i = event of interest
        # 2 if T_i <= tau and D_i = competing event
        event, duration, y_pred, ipcw = self._sort_by_duration(
            event, duration, y_pred, ipcw
        )
        # selects i such that T_i <= tau and D_i = 1 (event of interest)
        mask_event = event == 1
        y_pred_event, duration_event, ipcw_event = (
            y_pred[mask_event],
            duration[mask_event],
            ipcw[mask_event],
        )

        if self.remove_censoring:
            # For b type statistics, we only compare individuals who experienced
            # the event of interest with individuals who experienced a competing event.
            # selects j such that T_j <= tau and D_j = 2 (competing event)
            mask_competing = event == 2
            y_pred, duration, ipcw = (
                y_pred[mask_competing],
                duration[mask_competing],
                ipcw[mask_competing],
            )

        stats = Counter()
        # loop over i such that T_i <= tau and D_i = 1
        for y_pred_i, duration_i, ipcw_i in zip(
            y_pred_event, duration_event, ipcw_event
        ):
            # [idx_acceptable:] selects j such that the pair (i, j) is comparable
            # for type B, this means j such that (T_i >= T_j and D_j = 2)
            # for type A, this means j such that (T_i < T_j) or (T_i = T_j and D_j = 0)
            # n_ties_times returns the number of j such that (T_i = T_j and D_j = 1)
            idx_acceptable, n_ties_times = self._get_idx_acceptable(
                event, duration, duration_i
            )
            stats["n_ties_times"] += n_ties_times

            stats["n_pairs"] += duration.shape[0] - idx_acceptable
            stats["weighted_pairs"] += self._compute_weights(
                ipcw_i, ipcw[idx_acceptable:]
            )

            mask_ties_pred = np.abs(y_pred_i - y_pred[idx_acceptable:]) <= tied_tol
            stats["n_ties_pred"] += mask_ties_pred.sum()
            stats["weighted_ties_pred"] += self._compute_weights(
                ipcw_i, ipcw[idx_acceptable:][mask_ties_pred]
            )

            mask_concordant = y_pred_i > y_pred[idx_acceptable:]
            stats["n_concordant_pairs"] += (mask_concordant & ~mask_ties_pred).sum()
            stats["weighted_concordant_pairs"] += self._compute_weights(
                ipcw_i, ipcw[idx_acceptable:][mask_concordant & ~mask_ties_pred]
            )

        return stats

    def _preprocess_event(self, event, event_of_interest):
        """Map an event vector values to 0 (censoring), 1 (event of interest), \
            2 (competing risk)
        """
        event_out = np.full_like(event, 2)
        event_out[event == event_of_interest] = 1
        event_out[event == 0] = 0
        return event_out

    def _sort_by_duration(self, event, duration, y_pred, ipcw):
        """Sort the predictions and duration by the event duration.

        The pair type selects whether we sort duration by ascending or descending order.
        Indeed, to be comparable:
        - A pair of type A requires (T_i < T_j) | ((T_i = T_j) & (D_j = 0))
        - A pair of type B requires T_i >= T_j and D_j = 2
        """
        # For type A sort by ascending duration first, then by descending event.
        # After reordering, we would have:
        # duration = [10, 10, 10, 11]
        # event = [2, 1, 0, 1]
        # For type B sort by descending duration first, then by descending event.
        # After reordering, we would have:
        # duration = [11, 10, 10, 10]
        # event = [1, 2, 1, 0]
        indices = np.lexsort((-event, self.duration_sign*duration))
        event = event[indices]
        duration = duration[indices]
        y_pred = y_pred[indices]
        ipcw = ipcw[indices]

        return event, duration, y_pred, ipcw


class _StatsComputerTypeA(_StatsComputer):
    duration_sign = 1
    remove_censoring = False

    def _get_idx_acceptable(self, event, duration, duration_i):
        """Returns idx_acceptable and n_times_ties

        We select all acceptable type A pairs by keeping:
        - elements with `duration` strictly higher than `duration_i` (T_i < T_j) 
        - or censored elements at `duration_i` (T_i = T_j and D_j = 0).
        """
        # Using searchsorted with `side=right` returns the index of the smallest
        # element in `duration` strictly higher than `duration_i`.
        idx_strictly_higher = np.searchsorted(duration, duration_i, side="right")
        idx_with_ties = np.searchsorted(duration, duration_i, side="left")
        n_censored_ties = (event[idx_with_ties:idx_strictly_higher] == 0).sum()
        n_competing_ties = (event[idx_with_ties:idx_strictly_higher] == 2).sum()

        # +1 to remove the individual `i` from the ties count.
        start_idx = idx_with_ties + n_competing_ties + 1
        n_times_ties = (event[start_idx:idx_strictly_higher] == 1).sum()
        # `event` and `duration` sorted by ascending duration first and descending event second
        # [idx_acceptable:idx_strictly_higher] contains all j such that T_i = T_j and D_j = 0
        # [idx_strictly_higher:] contains all j such that T_i < T_j
        # [idx_acceptable:] contains all j such that (i, j) is a comparable type A pair
        idx_acceptable = idx_stricly_higher - n_censored_ties
        return idx_acceptable, n_times_ties

    def _compute_weights(self, ipcw_i, array_ipcw_j):
        """Compute W_{ij}^1 = G(T_i-|X_i) * G(T_i-|X_j)"""
        n_concordant_pairs = array_ipcw_j.shape[0]
        return n_concordant_pairs * (ipcw_i**2)


class _StatsComputerTypeB(_StatsComputer):
    duration_sign = -1
    remove_censoring = True

    def _get_idx_acceptable(self, event, duration, duration_i):
        """Returns idx_acceptable and n_times_ties

        We select all acceptable type B pairs by keeping elements with `duration` less than or equal to
        `duration_i` (T_i >= T_j and D_j = 2) as the duration has been filtered to only keep D_j = 2.
        """
        # `duration` sorted in descending order
        # Using searchsorted with `side=left` returns the index of the smallest element
        # in `duration` less than or equal to `duration_i`.
        # [idx_acceptable:] contains all j such that T_i >= T_j and D_j = 2
        idx_acceptable = np.searchsorted(-duration, -duration_i, side="left")
        return idx_acceptable, 0

    def _compute_weights(self, ipcw_i, array_ipcw_j):
        """Compute W_{ij}^2 = G(T_i-|X_i) * G(T_j-|X_j)."""
        return (ipcw_i * array_ipcw_j).sum()


def interpolate_preds(y_pred, time_grid, tau):
    """Interpolated the values of y_pred at tau."""
    tau = np.clip(tau, min(time_grid), max(time_grid))

    n_samples = y_pred.shape[0]
    y_pred_tau = np.zeros(n_samples)
    for idx in range(n_samples):
        y_pred_sample_at_tau = interp1d(
            x=time_grid,
            y=y_pred[idx, :],
            kind="linear",
        )(tau)
        y_pred_tau[idx] = y_pred_sample_at_tau

    return y_pred_tau
