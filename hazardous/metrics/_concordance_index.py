# %%
import warnings
from collections import Counter, defaultdict

import numpy as np
from scipy.interpolate import interp1d

from hazardous._ipcw import IPCWEstimator
from hazardous.utils import check_y_survival


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
    """Time-dependent concordance index for prognostic models \
    with competing risks with inverse probability of censoring weighting.

    The concordance index (C-index) is a common metric in survival analysis
    that evaluates if the model predictions correctly order pairs of individuals
    with respect to the order they actually experience the event of interest.
    It is defined as the probability that the prediction is concordant for a
    pair of individuals, and computed as the ratio between the number of
    concordant pairs and the total number of pairs. This implementation
    includes the extension to the competing events setting, where we consider
    multiple alternative events, and aim at determining which one happens first
    and when following the formulas and notations in [1].

    Due to the right-censoring in the data, the order of some pairs is unknown,
    so we define the notion of comparable pairs, i.e. the pairs for which
    we can compare the order of occurrence of the event of interest.
    A pair (i, j) is comparable, with i experiencing the event of interest
    at time T_i if:
    - j experiences the event of interest at a strictly greater time
        T_j > T_i (pair of type A)
    - j is censored at time T_j = T_i or greater (pair of type A)
    - j experiences a competing event before or at time T_i (pair of type B)
    The pair (i, j) is considered a tie for time if j experiences the event of
    interest at the same time (T_j=T_i). This tied time pair will be counted
    as `1/2` for the count of comparable pairs.
    A pair is then considered concordant if the incidence of the event of
    interest for `i` at time `tau` is larger than the incidence for `j`. If
    both predicted incidences are ties, the pair is counted as `1/2` for the
    count of concordant pairs.

    The c-index has been shown to depend on the censoring distribution, and an
    inverse probability of censoring weighting (IPCW) allows to overcome this
    limitation [2]. By default, the IPCW is implemented with a Kaplan-Meier
    estimator. Additionnally, the c-index is not a proper metric to evaluate
    a survival model [3], and alternatives as the integrated Brier score
    (`integrated_brier_score_incidence`) should be considered.

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
        Covariates, used to learn a censor model if the inverse
        probability of censoring weights (IPCW), if the IPCW model is conditional
        on features (for instance Cox). Not used if ipcw=None or 'km'.

    X_test: array or dataframe of shape (n_samples_test, n_features), default=None
        Covariates, used to predict weights a censor model if the inverse
        probability of censoring weights (IPCW), if the IPCW model is conditional
        on features (for instance Cox). Not used if ipcw=None or 'km'.

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
    cindex: (n_taus,)
        Value of the concordance index for each tau in taus.

    References
    ----------
    .. [1] Wolbers, M., Blanche, P., Koller, M. T., Witteman, J. C., &
           Gerds, T. A. (2014).
           Concordance for prognostic models with competing risks.
           Biostatistics, 15(3), 526-539.

    .. [2] Uno, H., Cai, T., Pencina, M. J., D'Agostino, R. B., &
           Wei, L. J. (2011).
           On the C‐statistics for evaluating overall adequacy of risk
           prediction procedures with censored survival data.
           Statistics in medicine, 30(10), 1105-1117.

    .. [3] Blanche, P., Kattan, M. W., & Gerds, T. A. (2019).
           The c-index is not proper for the evaluation of-year predicted risks.
           Biostatistics, 20(2), 347-357.

    .. [4] Gerds, T. A., Kattan, M. W., Schumacher, M., & Yu, C. (2013).
           Estimating a time‐dependent concordance index for survival
           prediction models with covariate dependent censoring.
           Statistics in medicine, 32(13), 2173-2184.

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
    return c_index_report["cindex"]


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

    All returned lists have the same length as taus.

    Returns
    -------
    cindex: list of float
        Value of the concordance index

    n_pairs_a: list of int
        Number of comparable pairs with T_i <= T_j (type A) without ties for D_j != 0
        those are the only comparable pairs for survival without competing events.

    n_concordant_pairs_a: list of int
        Number of concordant pairs among A pairs without ties.

    n_ties_times_a: list of int
        Number of tied pairs of type A with D_i = D_j = 1.

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
        ipcw_estimator_ = IPCWEstimator().fit(y_train)
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
    """Compute the cindex and the associated statistics for a given tau."""
    event, duration = check_y_survival(y_test)
    binary_event = (event == event_of_interest).astype("int32")
    if not binary_event.any():
        warnings.warn(
            f"There is not any event for {event_of_interest=!r}. "
            "The cindex is undefined."
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
    "Compute the C-index with a quadratic time complexity."
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
        # We select all acceptable pairs by only keeping elements strictly higher than
        # `duration`, which corresponds to A_{ij} = I(T_i < T_j).
        # Using `side=right` returns the index of the smallest element in `duration`
        # strictly higher than `duration_i`.
        idx_acceptable = np.searchsorted(duration, duration_i, side="right")
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

        stats["n_ties_times"] = (duration_event == duration_i).sum() - 1

    return stats


def _sort_by_duration(event, duration, y_pred, ipcw, pair_type):
    """Sort the predictions and duration by the event duration.

    The pair type selects whether we sort duration by ascending or descending order.
    Indeed, to be comparable:
    - A pair of type A requires T_i < T_j
    - A pair of type B requires T_i >= T_j
    """
    if pair_type == "b":
        duration *= -1
    indices = np.argsort(duration)
    event = event[indices]
    duration = duration[indices]
    y_pred = y_pred[indices]
    ipcw = ipcw[indices]

    return event, duration, y_pred, ipcw


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
