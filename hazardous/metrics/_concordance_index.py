# %%
import numpy as np
from lifelines import CoxPHFitter, KaplanMeierFitter
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
    """Time-dependent concordance index for prognostic models \
    with competing risks with covariate dependent censoring

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

    Returns
    -------


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

    .. [3] Wolbers, M., Blanche, P., Koller, M. T., Witteman, J. C.,
           & Gerds, T. A. (2014).
           Concordance for prognostic models with competing risks.
           Biostatistics, 15(3), 526-539.
    """
    # TODO: Check input parameters
    # - if 'km', then y_train must not be None
    # - if 'cox', the both y_train and X_train must not be None
    # - if tau is not None, then time_grid must not be None
    # check y, x

    y_test["event"] = (y_test["event"] == event_of_interest).astype("int")
    y_test["competing_event"] = (~y_test["event"].isin([0, event_of_interest])).astype(
        "int"
    )
    n_time_grid_test = y_test[y_test["event"] == 1].duration.nunique()

    if ipcw_estimator is not None and y_train is None:
        raise ValueError(
            f"ipcw_estimator is {ipcw_estimator}, but y_train is None."
            "Set y_train to use a IPCW estimator."
        )

    if ipcw_estimator is not None:
        # TODO: add cox option
        ipcw_estimator_ = IPCWEstimator().fit(y_train)
        # ipcw shape is (n_samples_test,)
        ipcw = ipcw_estimator_.compute_ipcw_at(y_test["duration"])

    else:
        if y_train is not None:
            # TODO: raise warning
            pass
        ipcw = np.ones(n_time_grid_test)

    # TODO: Bin y_test["duration"] using time grid?

    # Filter on tau
    c_index_scores = []
    for tau in taus:
        y_pred_tau = interpolate_preds(y_pred, time_grid, tau)
        c_index_scores.append(_concordance_index_tau(y_test, y_pred_tau, tau, ipcw))

    return c_index_scores


def _concordance_index_tau(y_test, y_pred_tau, tau, ipcw):
    y_test.loc[y_test["duration"] > tau, ["event", "competing_event"]] = 0

    num_correct_a, num_tied_a, num_pairs_a = _concordance_summary_statistics(
        y_test, y_pred_tau, ipcw
    )
    num_correct_b, num_tied_b, num_pairs_b = _concordance_summary_statistics(
        y_test, y_pred_tau, ipcw
    )

    numerator = num_correct_a + num_correct_b + 1 / 2 * (num_tied_a + num_tied_b)
    denominator = num_pairs_a + num_pairs_b

    return numerator / denominator


def _concordance_summary_statistics(
    y_test, y_pred, ipcw
):  # pylint: disable=too-many-locals
    """Find the concordance index in n * log(n) time.

    Assumes the data has been verified by lifelines.utils.concordance_index first.
    """
    # Here's how this works.
    #
    # It would be pretty easy to do if we had no censored data and no ties. There, the basic idea
    # would be to iterate over the cases in order of their true event time (from least to greatest),
    # while keeping track of a pool of *predicted* event times for all cases previously seen (= all
    # cases that we know should be ranked lower than the case we're looking at currently).
    #
    # If the pool has O(log n) insert and O(log n) RANK (i.e., "how many things in the pool have
    # value less than x"), then the following algorithm is n log n:
    #
    # Sort the times and predictions by time, increasing
    # n_pairs, n_correct := 0
    # pool := {}
    # for each prediction p:
    #     n_pairs += len(pool)
    #     n_correct += rank(pool, p)
    #     add p to pool
    #
    # There are three complications: tied ground truth values, tied predictions, and censored
    # observations.
    #
    # - To handle tied true event times, we modify the inner loop to work in *batches* of observations
    # p_1, ..., p_n whose true event times are tied, and then add them all to the pool
    # simultaneously at the end.
    #
    # - To handle tied predictions, which should each count for 0.5, we switch to
    #     n_correct += min_rank(pool, p)
    #     n_tied += count(pool, p)
    #
    # - To handle censored observations, we handle each batch of tied, censored observations just
    # after the batch of observations that died at the same time (since those censored observations
    # are comparable all the observations that died at the same time or previously). However, we do
    # NOT add them to the pool at the end, because they are NOT comparable with any observations
    # that leave the study afterward--whether or not those observations get censored.
    if np.logical_not(event_observed).all():
        return (0, 0, 0)

    died_mask = event_observed.astype(bool)
    # TODO: is event_times already sorted? That would be nice...
    died_truth = event_times[died_mask]
    ix = np.argsort(died_truth)
    died_truth = died_truth[ix]
    died_pred = predicted_event_times[died_mask][ix]

    censored_truth = event_times[~died_mask]
    ix = np.argsort(censored_truth)
    censored_truth = censored_truth[ix]
    censored_pred = predicted_event_times[~died_mask][ix]

    censored_ix = 0
    died_ix = 0
    times_to_compare = _BTree(np.unique(died_pred))
    num_pairs = np.int64(0)
    num_correct = np.int64(0)
    num_tied = np.int64(0)

    # we iterate through cases sorted by exit time:
    # - First, all cases that died at time t0. We add these to the sortedlist of died times.
    # - Then, all cases that were censored at time t0. We DON'T add these since they are NOT
    #   comparable to subsequent elements.
    while True:
        has_more_censored = censored_ix < len(censored_truth)
        has_more_died = died_ix < len(died_truth)
        # Should we look at some censored indices next, or died indices?
        if has_more_censored and (
            not has_more_died or died_truth[died_ix] > censored_truth[censored_ix]
        ):
            pairs, correct, tied, next_ix = _handle_pairs(
                censored_truth, censored_pred, censored_ix, times_to_compare
            )
            censored_ix = next_ix
        elif has_more_died and (
            not has_more_censored or died_truth[died_ix] <= censored_truth[censored_ix]
        ):
            pairs, correct, tied, next_ix = _handle_pairs(
                died_truth, died_pred, died_ix, times_to_compare
            )
            for pred in died_pred[died_ix:next_ix]:
                times_to_compare.insert(pred)
            died_ix = next_ix
        else:
            assert not (has_more_died or has_more_censored)
            break

        num_pairs += pairs
        num_correct += correct
        num_tied += tied

    return (num_correct, num_tied, num_pairs)


def _handle_pairs(truth, pred, first_ix, times_to_compare):
    """
    Handle all pairs that exited at the same time as truth[first_ix].

    Returns
    -------
      (pairs, correct, tied, next_ix)
      new_pairs: The number of new comparisons performed
      new_correct: The number of comparisons correctly predicted
      next_ix: The next index that needs to be handled
    """
    next_ix = first_ix
    while next_ix < len(truth) and truth[next_ix] == truth[first_ix]:
        next_ix += 1
    pairs = len(times_to_compare) * (next_ix - first_ix)
    correct = np.int64(0)
    tied = np.int64(0)
    for i in range(first_ix, next_ix):
        rank, count = times_to_compare.rank(pred[i])
        correct += rank
        tied += count

    return (pairs, correct, tied, next_ix)


def _sort_duration(y_test, y_pred):
    event, duration = check_y_survival(y_test)
    indices = np.argsort(duration)
    duration = duration[indices]
    y_pred = y_pred[indices]

    return event, duration, y_pred


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
import pandas as pd
from lifelines.datasets import load_leukemia
from lifelines.fitters.coxph_fitter import CoxPHFitter
from lifelines.utils.concordance import concordance_index

df = pd.read_csv("../../../lifelines/lifelines/datasets/anderson.csv", sep=" ")
cph = CoxPHFitter().fit(df, "t", "status")
concordance_index(df["t"], -cph.predict_partial_hazard(df), df["status"])


# %%
