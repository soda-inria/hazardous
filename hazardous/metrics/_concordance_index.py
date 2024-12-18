# %%
import warnings
from collections import Counter, defaultdict

import numpy as np
from scipy.interpolate import interp1d

from .._ipcw import KaplanMeierIPCW
from ..utils import check_y_survival


def concordance_index_incidence(
    y_test,
    y_pred,
    time_grid=None,
    taus=None,
    y_train=None,
    X_train=None,
    X_test=None,
    event_of_interest=1,
    ipcw_estimator="km",
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
        \tilde{N}^1_i(t) &= I\{\tilde{T} \leq t, \tilde{D}_i = 1\} \\
        \tilde{A}_{ij} &= I\{\tilde{T}_i < \tilde{T}_j\} \\
        \tilde{B}_{ij} &= I\{\tilde{T}_i \geq \tilde{T}_j, D_j = 2\} \\
        \hat{W}_{ij,1} &= \hat{G}(\tilde{T}_i-|X_i) \hat{G}(\tilde{T}_i|X_j) \\
        \hat{W}_{ij,2} &= \hat{G}(\tilde{T}_i-|X_i) \hat{G}(\tilde{T}_j-|X_j) \\
        Q_{ij}(t) &= I\{M(t, X_i) > M(t, X_j)\}
        \end{align}

    where :math:`D_j = 1` and :math:`D_j = 2` respectively indicate individuals
    having experienced the event of interest and a competing event, :math:`\hat{G}`
    is a IPCW estimator, and :math:`Q_{ij}(t)` is an indicator for the order of
    predicted risk at :math:`t`.

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

    y_pred: array of shape (n_samples_test, n_time_grid)
        Cumulative incidence for the event of interest, at the time points
        from the input time_grid.

    time_grid: array of shape (n_time_grid,), default=None
        Time points used to predict the cumulative incidence.

    taus: array of shape (n_taus,), default=None
        float or vector, timepoints at which the concordance index is
        evaluated.

    y_train : array, dictionnary or dataframe of shape (n_samples, 2)
        The train target, consisting in the 'event' and 'duration' columns.

    X_train: array or dataframe of shape (n_samples_train, n_features), default=None
        Covariates, used to learn a censor model if the inverse probability of
        censoring weights (IPCW) is conditional on features (for instance Cox).
        Unused if ipcw is None or 'km'.

    X_test: array or dataframe of shape (n_samples_test, n_features), default=None
        Covariates, used to predict weights a censor model if the inverse probability
        of censoring weights (IPCW) is conditional on features (for instance Cox).
        Unused if ipcw is None or 'km'.

    event_of_interest: int, default=1
        For competing risks, the event of interest.

    ipcw_estimator : {None or 'km'}, default="km"
        The inverse probability of censoring weighted (IPCW) estimator.
        - None set uniform weights to all samples
        - "km" use the Kaplan-Meier estimator

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
        time_grid=time_grid,
        taus=taus,
        y_train=y_train,
        X_train=X_train,
        X_test=X_test,
        event_of_interest=event_of_interest,
        ipcw_estimator=ipcw_estimator,
        tied_tol=tied_tol,
    )
    return np.array(c_index_report["cindex"])


def _concordance_index_incidence_report(
    y_test,
    y_pred,
    time_grid=None,
    taus=None,
    y_train=None,
    X_train=None,
    X_test=None,
    event_of_interest=1,
    ipcw_estimator="km",
    tied_tol=1e-8,
):
    """Report version of function `concordance_index_incidence`.

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

    n_ties_times_a: list of int
        Number of tied pairs of type A with (D_i = D_j = 1) & (T_i = T_j).

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
    binary_event = (event == event_of_interest).astype("int32")
    if not binary_event.any():
        warnings.warn(
            f"There is not any event for {event_of_interest=!r}. "
            "The C-index is undefined."
        )

    stats_a = _concordance_summary_statistics(
        event=binary_event,
        duration=duration,
        y_pred=y_pred,
        ipcw=ipcw,
        pair_type="a",
        tied_tol=tied_tol,
    )

    is_competing_event = (~np.isin(event, [0, event_of_interest])).any()
    if is_competing_event:
        mask_uncensored = event != 0
        binary_event = binary_event[mask_uncensored]
        duration = duration[mask_uncensored]
        y_pred = y_pred[mask_uncensored]
        ipcw = ipcw[mask_uncensored]

        stats_b = _concordance_summary_statistics(
            event=binary_event,
            duration=duration,
            y_pred=y_pred,
            ipcw=ipcw,
            pair_type="b",
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

    keys_b = ["n_pairs", "n_concordant_pairs", "n_ties_pred"]
    keys_a = keys_b + ["n_ties_times"]

    return {
        "cindex": cindex,
        **{f"{k}_a": stats_a[k] for k in keys_a},
        **{f"{k}_b": stats_b[k] for k in keys_b},
    }


def _concordance_summary_statistics(
    event, duration, y_pred, ipcw, pair_type, tied_tol=1e-8
):
    """Compute the C-index with a quadratic time complexity."""
    event, duration, y_pred, ipcw = _sort_by_duration(
        event, duration, y_pred, ipcw, pair_type=pair_type
    )
    mask_event = event == 1
    y_pred_event, duration_event, ipcw_event = (
        y_pred[mask_event],
        duration[mask_event],
        ipcw[mask_event],
    )
    if pair_type == "b":
        # For b type statistics, we only compare individuals who experienced the event
        # of interest with individuals who experienced a competing event.
        y_pred, duration, ipcw = (
            y_pred[~mask_event],
            duration[~mask_event],
            ipcw[~mask_event],
        )

    stats = Counter()
    for y_pred_i, duration_i, ipcw_i in zip(y_pred_event, duration_event, ipcw_event):
        idx_acceptable, n_ties_times = _get_idx_acceptable(
            event, duration, duration_i, pair_type
        )
        stats["n_ties_times"] += n_ties_times

        stats["n_pairs"] += duration.shape[0] - idx_acceptable
        stats["weighted_pairs"] += _compute_weights(
            ipcw_i, ipcw[idx_acceptable:], pair_type
        )

        mask_ties_pred = np.abs(y_pred_i - y_pred[idx_acceptable:]) <= tied_tol
        stats["n_ties_pred"] += mask_ties_pred.sum()
        stats["weighted_ties_pred"] += _compute_weights(
            ipcw_i, ipcw[idx_acceptable:][mask_ties_pred], pair_type
        )

        mask_concordant = y_pred_i > y_pred[idx_acceptable:]
        stats["n_concordant_pairs"] += (mask_concordant & ~mask_ties_pred).sum()
        stats["weighted_concordant_pairs"] += _compute_weights(
            ipcw_i, ipcw[idx_acceptable:][mask_concordant & ~mask_ties_pred], pair_type
        )

    return stats


def _sort_by_duration(event, duration, y_pred, ipcw, pair_type):
    """Sort the predictions and duration by the event duration.

    The pair type selects whether we sort duration by ascending or descending order.
    Indeed, to be comparable:
    - A pair of type A requires (T_i < T_j) | ((T_i = T_j) & (D_j = 0))
    - A pair of type B requires T_i >= T_j
    """
    if pair_type == "b":
        duration *= -1
    # Sort by ascending duration first, then by descending event.
    # After reordering, we would have:
    # duration = [10, 10, 10, 11]
    # event = [2, 1, 0, 1]
    indices = np.lexsort((-event, duration))
    event = event[indices]
    duration = duration[indices]
    y_pred = y_pred[indices]
    ipcw = ipcw[indices]

    return event, duration, y_pred, ipcw


def _get_idx_acceptable(event, duration, duration_i, pair_type):
    """Returns idx_acceptable and n_times_ties"""
    if pair_type == "a":
        # We select all acceptable pairs by only keeping elements strictly higher than
        # `duration`, which corresponds to A_{ij} = I(T_i < T_j).
        # We also select censored pairs where T_i = T_j and D_j = 0.
        # Using searchsorted with `side=right` returns the index of the smallest
        # element in `duration` strictly higher than `duration_i`.
        idx_acceptable = np.searchsorted(duration, duration_i, side="right")
        idx_with_ties = np.searchsorted(duration, duration_i, side="left")
        n_censored_ties = (event[idx_with_ties:idx_acceptable] == 0).sum()
        # +1 to remove the individual `i` from the ties count.
        n_times_ties = (event[idx_with_ties + 1 : idx_acceptable] == 1).sum()
        return idx_acceptable - n_censored_ties, n_times_ties
    else:
        # We select all acceptable pairs by keeping elements higher or equal than
        # `duration`, which corresponds to B_{ij} = I((T_j >= T_i)) & (D_j = 2))
        # as the duration has been filtered to only keep D_j = 2.
        # Using searchsorted with `side=left` returns the index of the smallest element
        # in `duration` higher or equal than `duration_i`.
        return np.searchsorted(duration, duration_i, side="left"), 0


def _compute_weights(ipcw_i, array_ipcw_j, pair_type):
    """Compute W_{ij}^1 when pair_type is a, and W_{ij}^2 when pair_type is b."""
    if pair_type == "a":
        n_concordant_pairs = array_ipcw_j.shape[0]
        return n_concordant_pairs * (ipcw_i**2)
    else:
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


# %%
