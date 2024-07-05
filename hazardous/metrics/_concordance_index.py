# %%
import warnings
import numpy as np
from collections import defaultdict
from scipy.interpolate import interp1d

from hazardous._ipcw import IPCWEstimator
from hazardous.metrics._btree import _BTree
from hazardous.utils import check_y_survival


def concordance_index_incidence(
    y_test,
    y_pred,
    time_grid=None,
    y_train=None,
    X_train=None,
    X_test=None,
    taus=None,
    event_of_interest=1,
    ipcw_estimator="km",
    tied_tol=1e-8,
):
    """Time-dependent concordance index for prognostic models \
    with competing risks with inverse probability of censoring weighting

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
    y_train : np.array, dictionnary or dataframe of shape (n_samples, 2)
        The train target, consisting in the 'event' and 'duration' columns.

    y_test : np.array, dictionnary or dataframe of shape (n_samples, 2)
        The test target, consisting in the 'event' and 'duration' columns.

    y_pred: (n_samples_test, n_time_grid)
        cumulative incidence for the event of interest, at the time points
        from the input time_grid

    time_grid: (n_time_grid,)
        time points used to predict the cumulative incidence

    X_train: (n_samples_train, n_features)
        covariates, used to learn a censor model if the inverse
        probability of censoring weights (IPCW), if the IPCW model is conditional
        on features (for instance cox). Not used if ipcw=None or 'km'.

    X_test: (n_samples_test, n_features)
        covariates, used to predict weights a censor model if the inverse
        probability of censoring weights (IPCW), if the IPCW model is conditional
        on features (for instance cox). Not used if ipcw=None or 'km'.

    taus: (n_taus,)
        float or vector, timepoints at which the concordance index is
        evaluated.

    event_of_interest: int
        For competing risks, the event of interest.

    Returns
    -------
    cindex: (n_taus,)
        value of the concordance index for each tau in taus.

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
        y_train=y_train,
        X_train=X_train,
        X_test=X_test,
        taus=taus,
        event_of_interest=event_of_interest,
        ipcw_estimator=ipcw_estimator,
        tied_tol=tied_tol,
    )
    return c_index_report["cindex"]


def _concordance_index_incidence_report(
    y_test,
    y_pred,
    time_grid=None,
    y_train=None,
    X_train=None,
    X_test=None,
    taus=None,
    event_of_interest=1,
    ipcw_estimator="km",
    tied_tol=1e-8,
):
    """
    Report version of function `concordance_index_incidence`.
    
    Returns
    -------
    cindex: float
        value of the concordance index

    num_pairs_a: integer
        number of comparable pairs with T_i <= T_j (type A) without ties for D_j != 0
        those are the only comparable pairs for survival without competing events

    num_concordant_pairs_a: integer
        number of concordant pairs among A pairs without ties

    num_tied_times_a: integer
        number of tied pairs of type A with D_i = D_j = 1

    num_tied_pred_a: integer
        number of pairs of type A with np.abs(y_pred_i - y_pred_i) < tied_tol

    num_pairs_b: integer
        number of comparable pairs with T_i >= T_j where j has a competing event (type B)
        returns 0 if there are no competing events

    num_concordant_pairs_b: integer
        number of concordant pairs among B pairs without ties

    num_tied_pred_b: integer
        number of pairs of type B with np.abs(y_pred_i - y_pred_i) < tied_tol   

    """
    # TODO: Check input parameters
    # - if tau is not None, then time_grid must not be None
    # check y, x
    y_test["competing_event"] = (~y_test["event"].isin([0, event_of_interest])).astype(
        "int"
    )
    y_test["event"] = (y_test["event"] == event_of_interest).astype("int")
    is_competing_event = y_test["competing_event"].any()

    n_time_grid_test = y_test[y_test["event"] == 1]["duration"].nunique()

    if ipcw_estimator is not None:
        # TODO: add cox option
        ipcw_estimator_ = IPCWEstimator().fit(y_train)
        ipcw = ipcw_estimator_.compute_ipcw_at(y_test["duration"]) # shape: (n_samples_test,)

    else:
        if y_train is not None:
            warnings.warns(
                "y_train passed but ipcw_estimator is set to None, "
                "therefore y_train won't be used. Set y_train=None "
                "to silence this warning."
            )
        ipcw = np.ones(n_time_grid_test)

    # XXX: Bin y_test["duration"] using time grid?

    c_index_report = defaultdict(list)
    
    for tau in taus:
        y_pred_tau = interpolate_preds(y_pred, time_grid, tau)
        c_index_report_tau = _concordance_index_tau(
            y_test, y_pred_tau, tau, ipcw, is_competing_event
        )
        for metric_name, metric_value in c_index_report_tau.items():
            c_index_report[metric_name].append(metric_value)

    return c_index_report


def _concordance_index_tau(y_test, y_pred_tau, tau, ipcw, is_competing_event):
    y_test = y_test.copy()
    y_test.loc[y_test["duration"] > tau, ["event"]] = 0

    stats_pairs_a = _concordance_summary_statistics(
        event=y_test["event"],
        duration=y_test["duration"],
        y_pred=y_pred_tau,
        ipcw=ipcw,
    )

    weighted_corrects = stats_pairs_a["weighted_corrects"]
    weighted_ties = stats_pairs_a["weighted_ties"]
    weighted_pairs = stats_pairs_a["weighted_pairs"]

    stats_pairs_b = {}
    if is_competing_event:
        mask_uncensored = y_test["event"] != 0
        y_test = y_test.loc[mask_uncensored]
        y_pred_tau = y_pred_tau[mask_uncensored]
        ipcw = ipcw[mask_uncensored]

        stats_pairs_b = _concordance_summary_statistics(
            event=y_test["competing_event"],
            duration=y_test["duration"],
            y_pred=y_pred_tau,
            ipcw=ipcw,
        )
        
        weighted_corrects += stats_pairs_b["weighted_corrects"]
        weighted_ties += stats_pairs_b["weighted_ties"]
        weighted_pairs += stats_pairs_b["weighted_pairs"]

    cindex = (weighted_corrects + .5 * (weighted_ties)) / weighted_pairs

    return dict(
        cindex=cindex,
        # a statistics
        num_pairs_a=stats_pairs_a["num_pairs"],
        num_concordant_pairs_a=stats_pairs_a["num_concordant_pairs"],
        num_tied_times_a=stats_pairs_a["num_tied_times"],
        num_tied_pred_a=stats_pairs_a["num_tied_pred"],
        # b statistics
        num_pairs_b=stats_pairs_b.get("num_pairs", 0),
        num_concordant_pairs_b=stats_pairs_b.get("num_concordant_pairs", 0),
        num_tied_pred_b=stats_pairs_b.get("num_tied_pred", 0),
    )


def _concordance_summary_statistics(event, duration, y_pred, ipcw):
    """Find the concordance index in n * log(n) time.

    Here's how this works.
    
    It would be pretty easy to do if we had no censored data and no ties. There, the basic idea
    would be to iterate over the cases in order of their true event time (from least to greatest),
    while keeping track of a pool of *predicted* event times for all cases previously seen (= all
    cases that we know should be ranked lower than the case we're looking at currently).
    
    If the pool has O(log n) insert and O(log n) RANK (i.e., "how many things in the pool have
    value less than x"), then the following algorithm is n log n:
    
    Sort the times and predictions by time, increasing
    n_pairs, n_correct := 0
    pool := {}
    for each prediction p:
        n_pairs += len(pool)
        n_correct += rank(pool, p)
        add p to pool
    
    There are three complications: tied ground truth values, tied predictions, and censored
    observations.
    
    - To handle tied true event times, we modify the inner loop to work in *batches* of observations
    p_1, ..., p_n whose true event times are tied, and then add them all to the pool
    simultaneously at the end.
    
    - To handle tied predictions, which should each count for 0.5, we switch to
        n_correct += min_rank(pool, p)
        n_tied += count(pool, p)
    
    - To handle censored observations, we handle each batch of tied, censored observations just
    after the batch of observations that died at the same time (since those censored observations
    are comparable all the observations that died at the same time or previously). However, we do
    NOT add them to the pool at the end, because they are NOT comparable with any observations
    that leave the study afterward--whether or not those observations get censored.
    """
    event_mask = event.astype(bool)

    # Sort the predictions by the event duration
    duration_event = duration[event_mask]
    indices = np.argsort(duration_event)
    duration_event = duration_event[indices]
    y_pred_event = y_pred[event_mask][indices]

    # Sort the predictions by the censoring duration
    duration_censoring = duration[~event_mask]
    indices = np.argsort(duration_censoring)
    duration_censoring = duration_censoring[indices]
    y_pred_censoring = y_pred_event[~event_mask][indices]

    tree_pred = _BTree(
        nodes=np.unique(y_pred_event),
        weights=ipcw,
    )

    idx_event = idx_censoring = 0
    num_pairs = num_correct = num_tied = 0

    # We iterate through test samples sorted by duration:
    # - First, all cases that died at t0. We add these to the btree.
    # - Then, all cases that were censored at t0. We don't add these since they are
    #   not comparable to subsequent elements.
    has_censored_left = idx_censoring < len(duration_censoring)
    has_event_left = idx_event < len(duration_event)
    
    while has_event_left or has_censored_left:
        
        is_event_before_censoring = (
            duration_event[idx_event] <= duration_censoring[idx_censoring]
        )
        if has_event_left and (is_event_before_censoring or not has_censored_left):
            pairs, correct, tied_prediction, tied_duration = _handle_pairs(
                duration_event, y_pred_event, idx_event, tree_pred
            )
            for _ in range(tied_duration):
                tree_pred.insert(y_pred_event[idx_event], idx_event)
                idx_event += 1
 
        else:
            pairs, correct, tied_prediction, tied_duration = _handle_pairs(
                duration_censoring, y_pred_censoring, idx_censoring, tree_pred
            )
            idx_censoring += tied_duration
        
        num_pairs += pairs
        num_correct += correct
        num_tied += tied_prediction

        has_event_left = idx_event < len(duration_event)
        has_censored_left = idx_censoring < len(duration_censoring)

    return (num_correct, num_tied, num_pairs)


def _handle_pairs(duration, y_pred, idx, tree_pred):
    """
    Handle all pairs that exited at the same duration[idx].

    Returns
    -------
      (pairs, correct, tied, next_ix)
      new_pairs: The number of new comparisons performed
      new_correct: The number of comparisons correctly predicted
      next_ix: The next index that needs to be handled
    """
    tied_duration = 0
    while (
        idx + tied_duration < len(duration)
        and duration[idx] == duration[idx + tied_duration]
    ):
        tied_duration += 1

    pairs = tree_pred.total_counts(idx) * tied_duration
    
    correct, tied_prediction = 0, 0
    for _ in range(tied_duration):
        rank, count = tree_pred.rank(
            value=y_pred[idx],
            idx_value=idx,
        )
        idx += 1
        correct += rank
        tied_prediction += count

    return (pairs, correct, tied_prediction, tied_duration)


def interpolate_preds(y_pred, time_grid, tau):
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
