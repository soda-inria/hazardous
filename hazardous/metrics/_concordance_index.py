# %%
import warnings
from collections import Counter, defaultdict

import numpy as np
from scipy.interpolate import interp1d

from hazardous._ipcw import IPCWEstimator
from hazardous.metrics._btree import _BTree
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
    and when following the formulas and notations in [3].

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
    limitation [1]. By default, the IPCW is implemented with a Kaplan-Meier
    estimator. Additionnally, the c-index is not a proper metric to evaluate
    a survival model [4], and alternatives as the integrated Brier score
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
        on features (for instance cox). Not used if ipcw=None or 'km'.

    X_test: array or dataframe of shape (n_samples_test, n_features), default=None
        Covariates, used to predict weights a censor model if the inverse
        probability of censoring weights (IPCW), if the IPCW model is conditional
        on features (for instance cox). Not used if ipcw=None or 'km'.

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
    .. [1] Uno, H., Cai, T., Pencina, M. J., D'Agostino, R. B., &
           Wei, L. J. (2011).
           On the C‐statistics for evaluating overall adequacy of risk
           prediction procedures with censored survival data.
           Statistics in medicine, 30(10), 1105-1117.

    .. [2] Gerds, T. A., Kattan, M. W., Schumacher, M., & Yu, C. (2013).
           Estimating a time‐dependent concordance index for survival
           prediction models with covariate dependent censoring.
           Statistics in medicine, 32(13), 2173-2184.

    .. [3] Wolbers, M., Blanche, P., Koller, M. T., Witteman, J. C., &
           Gerds, T. A. (2014).
           Concordance for prognostic models with competing risks.
           Biostatistics, 15(3), 526-539.

    .. [4] Blanche, P., Kattan, M. W., & Gerds, T. A. (2019).
           The c-index is not proper for the evaluation of-year predicted risks.
           Biostatistics, 20(2), 347-357.
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

    num_pairs_a: list of int
        Number of comparable pairs with T_i <= T_j (type A) without ties for D_j != 0
        those are the only comparable pairs for survival without competing events.

    num_concordant_pairs_a: list of int
        Number of concordant pairs among A pairs without ties.

    num_ties_times_a: list of int
        Number of tied pairs of type A with D_i = D_j = 1.

    num_ties_pred_a: list of int
        Number of pairs of type A with np.abs(y_pred_i - y_pred_j) < tied_tol

    num_pairs_b: list of int
        Number of comparable pairs with T_i >= T_j where j has
        a competing event (type B).
        0 if there are no competing events.

    num_concordant_pairs_b: list of int
        Number of concordant pairs among B pairs without ties.

    num_ties_pred_b: list of int
        Number of pairs of type B with np.abs(y_pred_i - y_pred_j) < tied_tol.
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

    # XXX: When using cox, Bin y_test["duration"] using time grid?

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

    for tau in taus:
        y_pred_tau = interpolate_preds(y_pred, time_grid, tau)
        y_test_tau = y_test.copy()
        y_test_tau.loc[y_test_tau["duration"] > tau, "event"] = 0
        # TODO: use tied_tol
        c_index_report_tau = _concordance_index_tau(
            y_test_tau, y_pred_tau, ipcw, event_of_interest
        )
        for metric_name, metric_value in c_index_report_tau.items():
            c_index_report[metric_name].append(metric_value)

    return c_index_report


def _concordance_index_tau(y_test, y_pred, ipcw, event_of_interest):
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
        )
    else:
        stats_b = Counter()

    stats = stats_a + stats_b

    if stats["weighted_pairs"] == 0:
        stats["weighted_pairs"] = np.nan

    cindex = (
        stats["weighted_concordant_pairs"] + 0.5 * stats["weighted_ties_pred"]
    ) / stats["weighted_pairs"]

    keys_b = ["num_pairs", "num_concordant_pairs", "num_ties_pred"]
    keys_a = keys_b + ["num_ties_times"]

    return {
        "cindex": cindex,
        **{f"{k}_a": stats_a[k] for k in keys_a},
        **{f"{k}_b": stats_b[k] for k in keys_b},
    }


def _concordance_summary_statistics(event, duration, y_pred, ipcw, pair_type="a"):
    """Find the concordance index in n * log(n) time.

    This function iterates over the samples ordered by ascending time to event,
    while keeping track of a tree of predicted probabilities for all cases
    previously seen, i.e. all cases that should be ranked lower than the case
    we're looking at currently.

    If the tree has:
    * O(log n) time complexity to insert
    * O(log n) time complexity to rank (i.e., "how many values in the
    tree are lower than x")

    then the following algorithm has a O(n * log n) time complexity:

    Sort the times and predictions by time, increasing
    n_pairs, n_correct := 0
    tree := {}
    for each prediction p:
        n_pairs += len(tree)
        n_correct += rank(tree, p)
        add p to tree

    There are three complications: tied ground duration, tied predictions,
    and censored observations.

    1. To handle tied duration, we modify the inner loop to work in *batches*
    of observations p_1, ..., p_n whose duration are tied, and then add them all
    to the pool simultaneously at the end.

    2. To handle tied predictions, which should each count for 0.5, we switch to
        n_correct += min_rank(tree, p)
        n_tied += count(tree, p)

    3. To handle censored observations, we handle each batch of tied,
    censored observations just after the batch of observations that died at
    the same time (since those censored observations are comparable all
    the observations that died at the same time or previously).
    However, we do not add them to the tree at the end, because they are not
    comparable with any observations that leave the study afterward,
    whether or not those observations get censored.
    """
    # For each node i, the balanced tree places:
    # - a lower predicted probability p_i at the node 2 * i + 1
    # - a higher predicted probability p_i at the node 2 * i + 2

    # For a given node to rank j, the balanced tree returns the number of inserted
    # values that are lower than p_j, i.e. the size of the left subtree of the node
    # whose value is p_j.

    # When p_j is high, we actually want the size of the left subtree to be small,
    # because it represents the number of concordant pairs, where:
    # * T_i < T_j, p_i > p_j and D_i = 1 (concordant pairs of type A)
    # * T_i > T_j, p_i > p_j and D_i = 1, D_j = 2 (concordant pairs of type B)

    # So, we build the balanced tree with the negative predicted probabilities of
    # incidence, ordered in ascending order. Note that `np.unique` sorts the unique
    # values in ascending order.
    y_pred = y_pred * -1

    y_pred_event, duration_event, ipcw_event = _sort_by_duration(
        event, duration, y_pred, ipcw, event_label=1, pair_type=pair_type
    )
    y_pred_censoring, duration_censoring, ipcw_censoring = _sort_by_duration(
        event, duration, y_pred, ipcw, event_label=0, pair_type=pair_type
    )
    unique_neg_preds = np.unique(y_pred_event)
    tree_pred = _BTree(
        nodes=unique_neg_preds,
        left_weights=ipcw_event,
        right_weights=ipcw_censoring,
    )

    idx_event = idx_censoring = 0
    total_stats = Counter()

    has_censored_left = idx_censoring < len(duration_censoring)
    has_event_left = idx_event < len(duration_event)

    while has_event_left or has_censored_left:
        is_event_before_censoring = has_event_left and (
            not has_censored_left
            or duration_event[idx_event] <= duration_censoring[idx_censoring]
        )
        if is_event_before_censoring:
            num_ties_times = _get_ties_times(duration_event, idx_event)
            stats = Counter({"num_ties_times": num_ties_times - 1})

            # Pairs of type B are only comparable when D_i = 1 and D_j = 2.
            # At this stage, we are comparing pairs with D_i = 1 and D_j = 1.
            # Therefore, we skip this operation for type B pairs.
            if pair_type == "a":
                stats += _handle_pairs(
                    num_ties_times,
                    y_pred_event,
                    idx_event,
                    tree_pred,
                    pair_type,
                )
            for _ in range(num_ties_times):
                tree_pred.insert(y_pred_event[idx_event], idx_event)
                idx_event += 1

        else:
            num_ties_times = _get_ties_times(duration_censoring, idx_censoring)
            stats = Counter({"num_ties_times": num_ties_times - 1})
            stats += _handle_pairs(
                num_ties_times,
                y_pred_censoring,
                idx_censoring,
                tree_pred,
                pair_type,
            )
            idx_censoring += num_ties_times

        total_stats += stats

        has_event_left = idx_event < len(duration_event)
        has_censored_left = idx_censoring < len(duration_censoring)

    return total_stats


def _get_ties_times(duration, jdx):
    """Count the number of identical duration for a given index jdx.

    Note that duration is a sorted array.
    """
    ties_duration = 0
    while (
        jdx + ties_duration < len(duration)
        and duration[jdx] == duration[jdx + ties_duration]
    ):
        ties_duration += 1

    return ties_duration


def _handle_pairs(ties_duration, y_pred, jdx, tree_pred, pair_type):
    """
    Handle all pairs that exited at the same duration[idx].

    Returns
    -------
    stats : Counter with the keys:
        - num_pairs
        - num_concordant_pairs
        - num_ties_pred
        - weighted_pairs
        - weighted_concordant_pairs
        - weighted_ties_pred
    """
    stats = Counter()

    use_left_weight_only = pair_type == "a"

    # Compute the number of pairs (num_pairs, weighted_pairs)
    for _ in range(ties_duration):
        stats += tree_pred.total_counts(
            jdx_right_weight=jdx,
            use_left_weight_only=use_left_weight_only,
            return_weighted=True,
        )

    # Compute the number of concordant pairs and ties
    for _ in range(ties_duration):
        rank, ties = tree_pred.rank(
            value=y_pred[jdx],
            jdx_right_weight=jdx,
            use_left_weight_only=use_left_weight_only,
            return_weighted=True,
        )
        stats["num_concordant_pairs"] += rank["num_pairs"]
        stats["weighted_concordant_pairs"] += rank["weighted_pairs"]
        stats["num_ties_pred"] += ties["num_pairs"]
        stats["weighted_ties_pred"] += ties["weighted_pairs"]

        jdx += 1

    return stats


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


def _sort_by_duration(event, duration, y_pred, ipcw, event_label, pair_type):
    """Sort the predictions and duration by the event duration.

    The pair type selects whether we sort duration by ascending or descending order.
    Indeed, to be comparable:
    - A pair of type A requires T_i < T_j
    - A pair of type B requires T_i >= T_j
    """
    event_mask = event == event_label
    duration = duration[event_mask]
    if pair_type == "b":
        duration *= -1
    indices = np.argsort(duration)
    duration = duration[indices]
    y_pred = y_pred[event_mask][indices]
    ipcw = ipcw[event_mask][indices]

    return y_pred, duration, ipcw
