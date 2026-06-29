import numpy as np

from .._km_sampler import _AalenJohansenSampler
from ..utils import check_y_survival
from ._km_calibration import km_calibration


def _truncation_mask(duration, times, min_prop_at_risk):
    """Boolean mask of times to keep for the calibration integral.

    Keeps the timepoints where the proportion of the set still at risk
    (i.e. with ``duration >= t``) is at least ``min_prop_at_risk``. This
    discards the noisy tail of the time grid where only a handful of
    subjects remain.
    """
    prop_at_risk = (np.asarray(duration)[:, None] >= np.asarray(times)[None, :]).mean(
        axis=0
    )
    return prop_at_risk >= min_prop_at_risk


def aj_calibration_at_t(y_calibration, times, pred_calibration, event_of_interest=None):
    r"""Pointwise AJ calibration error at each time point.

    For each event :math:`k`, computes the difference between the mean
    predicted CIF and the marginal Aalen-Johansen CIF at every time in
    ``times``:

    .. math::

        AJ_k(t) = |\bar{F}_k(t) - \hat{F}^{AJ}_k(t)|

    where :math:`\bar{F}_k(t) = \frac{1}{n} \sum_{i=1}^n \hat{F}_k(t \mid
    \mathbf{x}_i)` is the mean predicted cumulative incidence for event
    :math:`k` across the calibration set, and :math:`\hat{F}^{AJ}_k(t)`
    is the marginal Aalen-Johansen CIF for event :math:`k` fitted on the
    same set. The survival probability (event 0) is compared against the
    Kaplan-Meier estimate via :func:`km_calibration`.

    Parameters
    ----------
    y_calibration : array-like of shape (n_samples, 2)
        Survival outcomes of the calibration set, with columns
        ``"event"`` (0 for censoring, positive integers for each cause of
        event) and ``"duration"`` (observed time).

    times : array-like of shape (n_times,)
        Time points at which the CIFs were predicted. Need not be sorted;
        the last axis of ``pred_calibration`` must share the same ordering.

    pred_calibration : array-like of shape (n_samples, n_events+1, n_times)
        Predicted incidence probabilities at ``times`` for the calibration
        set. The second axis is indexed by event identifier in sorted
        order: index 0 holds the survival probability, indices 1, 2, …
        hold cause-specific CIFs.

    event_of_interest : int or None, default=None
        If provided, return only the difference array for that event.
        If ``None``, return a dict with one array per event.

    Returns
    -------
    differences : dict of {int: ndarray of shape (n_times,)}
        Pointwise difference :math:`AJ_k(t)` for each event identifier,
        in ascending time order.  Only the entry for ``event_of_interest``
        is returned when that parameter is set.

    See Also
    --------
    aj_calibration_per_event : Integrate these differences into a scalar
        score per event.
    km_calibration : KM-based calibration used for event 0.

    References
    ----------
    .. [Alberge2026] J. Alberge, T. Haugomat, G. Varoquaux, J. Abecassis,
       "On the calibration of survival models with competing risks",
       arXiv:2602.00194, 2026. https://arxiv.org/pdf/2602.00194
    """
    times = np.asarray(times)
    pred_calibration = np.asarray(pred_calibration)

    order = np.argsort(times)
    times = times[order]
    pred_calibration = pred_calibration[:, :, order]

    event, _ = check_y_survival(y_calibration)
    event_ids = np.array(sorted(set([0]) | set(event)))

    # Event 0: compare mean survival prediction against Kaplan-Meier
    _, diff_km = km_calibration(
        y_calibration, times, pred_calibration[:, 0, :], return_diff_at_t=True
    )
    differences = {0: np.abs(diff_km)}

    # Events 1..K: compare mean CIF prediction against Aalen-Johansen
    aalen_sampler = _AalenJohansenSampler().fit(y_calibration)
    for event_id in event_ids[1:]:
        inc_probs_aj = aalen_sampler.incidence_func_[event_id](times)
        inc_probs_mean = pred_calibration[:, event_id, :].mean(axis=0)
        differences[event_id] = np.abs(inc_probs_mean - inc_probs_aj)

    if event_of_interest is not None:
        return differences[event_of_interest]
    return differences


def aj_calibration_per_event(
    y_calibration,
    times,
    pred_calibration,
    event_of_interest=None,
    alpha=2,
    min_prop_at_risk=0.05,
):
    r"""AJ calibration score per event, integrated over time.

    Integrates the squared (or :math:`\alpha`-th power) pointwise
    calibration error over time for each event:

    .. math::

        \text{AJ-Cal}_k = \frac{1}{t_{\max}}
        \int_0^{t_{\max}} AJ_k(t)^\alpha \, dt

    where :math:`AJ_k(t) = |\bar{F}_k(t) - \hat{F}^{AJ}_k(t)|` is computed by
    :func:`aj_calibration_at_t`. Here :math:`\bar{F}_k(t) = \frac{1}{n}
    \sum_{i=1}^n \hat{F}_k(t \mid \mathbf{x}_i)` is the mean predicted
    cumulative incidence for event :math:`k` across the calibration set,
    :math:`\hat{F}^{AJ}_k(t)` is the marginal Aalen-Johansen CIF fitted on
    the same set, and the survival probability (event 0) is compared
    against the Kaplan-Meier estimate via :func:`km_calibration`.

    A score of zero indicates perfect marginal calibration for event
    :math:`k`.

    Parameters
    ----------
    y_calibration : array-like of shape (n_samples, 2)
        Survival outcomes of the calibration set, with columns
        ``"event"`` and ``"duration"``.

    times : array-like of shape (n_times,)
        Time points at which the CIFs were predicted.

    pred_calibration : array-like of shape (n_samples, n_events+1, n_times)
        Predicted incidence probabilities at ``times`` for the calibration
        set.

    event_of_interest : int or None, default=None
        If provided, return only the score for that event.
        If ``None``, return a dict with one score per event.

    alpha : int, default=2
        Exponent applied to :math:`AJ_k(t)` before integration.
        ``alpha=2`` gives a squared L2 calibration score; ``alpha=1``
        gives the L1 score.

    min_prop_at_risk : float, default=0.05
        Lower bound on the proportion of the set still at risk required
        to include a timepoint in the integral. The integration stops once
        fewer than this fraction of subjects remain at risk, which avoids
        measuring noise in the tail of the time grid where the
        Aalen-Johansen reference is unreliable. Set to ``0`` to integrate
        over the full time grid.

    Returns
    -------
    scores : dict of {int: float} or float
        Integrated calibration score for each event. Returns a single
        float when ``event_of_interest`` is set.

    See Also
    --------
    aj_calibration_at_t : Pointwise differences used in the integration.
    aj_calibration : Aggregate all per-event scores into one number.

    References
    ----------
    .. [Alberge2026] J. Alberge, T. Haugomat, G. Varoquaux, J. Abecassis,
       "On the calibration of survival models with competing risks",
       arXiv:2602.00194, 2026. https://arxiv.org/pdf/2602.00194
    """
    times = np.asarray(times)
    order = np.argsort(times)
    times_sorted = times[order]
    inc_sorted = np.asarray(pred_calibration)[:, :, order]

    _, duration = check_y_survival(y_calibration)
    mask = _truncation_mask(duration, times_sorted, min_prop_at_risk)
    if not mask.any():
        raise ValueError(
            f"min_prop_at_risk={min_prop_at_risk} leaves no timepoints; "
            "lower it or pass 0."
        )
    times_sorted = times_sorted[mask]
    inc_sorted = inc_sorted[:, :, mask]

    t_max = times_sorted[-1]
    differences = aj_calibration_at_t(y_calibration, times_sorted, inc_sorted)

    scores = {
        event_id: np.trapezoid(diff**alpha, times_sorted) / t_max
        for event_id, diff in differences.items()
    }

    if event_of_interest is not None:
        return scores[event_of_interest]
    return scores


def aj_calibration(
    y_calibration,
    times,
    pred_calibration,
    alpha=2,
    reduction="mean",
    min_prop_at_risk=0.05,
):
    r"""Overall AJ calibration score aggregated across all events.

    Computes the per-event AJ calibration scores via
    :func:`aj_calibration_per_event`, then reduces them to a single number:

    .. math::

        \text{AJ-Cal} = \frac{1}{K} \sum_{k=1}^{K} \text{AJ-Cal}_k
        \quad \text{(reduction='mean')}

    or the sum (resp. max) when ``reduction='sum'`` (resp. ``reduction='max'``).

    Each per-event score integrates the pointwise error
    :math:`AJ_k(t) = |\bar{F}_k(t) - \hat{F}^{AJ}_k(t)|` between the mean
    predicted cumulative incidence :math:`\bar{F}_k(t) = \frac{1}{n}
    \sum_{i=1}^n \hat{F}_k(t \mid \mathbf{x}_i)` across the calibration
    set and the marginal Aalen-Johansen reference
    :math:`\hat{F}^{AJ}_k(t)` fitted on the same set (Kaplan-Meier via
    :func:`km_calibration` for event 0).

    Parameters
    ----------
    y_calibration : array-like of shape (n_samples, 2)
        Survival outcomes of the calibration set, with columns
        ``"event"`` and ``"duration"``.

    times : array-like of shape (n_times,)
        Time points at which the CIFs were predicted.

    pred_calibration : array-like of shape (n_samples, n_events+1, n_times)
        Predicted incidence probabilities at ``times`` for the calibration
        set.

    alpha : int, default=2
        Exponent applied to the pointwise difference before integration.

    reduction : {"mean", "sum", "max"}, default="mean"
        How to aggregate per-event scores into a single value.

    min_prop_at_risk : float, default=0.05
        Lower bound on the proportion of the set still at risk required
        to include a timepoint in the integral. Stops the integration once
        fewer than this fraction of subjects remain at risk. Set to ``0``
        to integrate over the full time grid.

    Returns
    -------
    score : float
        Aggregated AJ calibration score.

    See Also
    --------
    aj_calibration_per_event : Per-event scores before aggregation.
    aj_calibration_at_t : Pointwise calibration error at each time point.
    km_calibration : KM-Calibration for single-event survival.

    References
    ----------
    .. [Alberge2026] J. Alberge, T. Haugomat, G. Varoquaux, J. Abecassis,
       "On the calibration of survival models with competing risks",
       arXiv:2602.00194, 2026. https://arxiv.org/pdf/2602.00194
    """
    if reduction not in ("mean", "sum", "max"):
        raise ValueError(
            f"reduction must be 'max', 'mean' or 'sum', got {reduction!r}."
        )

    scores = aj_calibration_per_event(
        y_calibration,
        times,
        pred_calibration,
        alpha=alpha,
        min_prop_at_risk=min_prop_at_risk,
    )
    values = np.array(list(scores.values()))

    if reduction == "mean":
        return float(np.mean(values))
    elif reduction == "max":
        return float(np.max(values))
    return float(np.sum(values))
