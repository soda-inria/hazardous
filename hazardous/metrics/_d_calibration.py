import numpy as np

from ..utils import check_y_survival


def d_calibration(
    fk,
    fk_infty,
    s_t,
    y_conf,
    event_of_interest="any",
    epsilon=1e-3,
    n_buckets=100,
    times=None,
):
    r"""Compute per-bucket DCR-calibration values.

    Implements Definition 3.2 from the paper: compute b̂_k[0,ρ] for each
    bucket ρ ∈ [0,1]. These calibration values measure how well the
    predicted marginal cumulative incidence aligns with observed outcomes
    across different predicted risk levels.

    For each bucket [0, ρ], counts:
    - Events (δᵢ = k) where the normalized prediction F̂_k(tᵢ|xᵢ) / F̂_k(∞|xᵢ) ∈ [0, ρ]
    - For censored observations (δᵢ = 0), the contribution:
      (ρ * F̂_k(∞|xᵢ) - F̂_k(tᵢ|xᵢ)) / S(tᵢ|xᵢ)

    The returned cumulative distribution b̂_k[0,ρ] compares predicted
    risk levels with observed event frequencies.

    Parameters
    ----------
    fk : array-like
        If times is None (exact approach):
            shape (n_samples,), CIF for event k evaluated at observed times.
        If times is not None (interpolation approach):
            shape (n_samples, n_times), CIF on a time grid. Will be interpolated
            to each individual's observed time.

    fk_infty : array-like
        If times is None (exact approach):
            shape (n_samples,), marginal event probability.
        If times is not None (interpolation approach):
            shape (n_samples, n_times), marginal probabilities on the time grid.
            Will be interpolated to each individual's observed time.

    s_t : array-like
        If times is None (exact approach):
            shape (n_samples,), survival at observed times.
        If times is not None (interpolation approach):
            shape (n_samples, n_times), survival on the time grid.
            Will be interpolated to each individual's observed time.

    y_conf : array-like of shape (n_samples, 2)
        Survival outcomes with columns "event" (0 for censoring,
        positive integers for each cause) and "duration" (observed time).

    event_of_interest : {"any", int}, default="any"
        Which event to compute calibration for.
        - "any": any event occurred (δᵢ > 0)
        - int: specific event type (δᵢ = event_of_interest)

    epsilon : float, default=1e-3
        Small constant added to denominators for numerical stability.

    n_buckets : int, default=100
        Number of equiprobable buckets for the calibration histogram.
        Creates buckets at quantiles [0, 1/n_buckets, 2/n_buckets, ..., 1].

    times : array-like or None, default=None
        Time grid at which predictions are evaluated.
        If None, use exact approach: fk, fk_infty, s_t are 1D arrays.
        If not None, use interpolation: fk, fk_infty, s_t are 2D arrays
        evaluated on the time grid, and will be interpolated to each
        individual's observed time.

    Returns
    -------
    calibration : ndarray of shape (n_buckets,)
        Cumulative calibration histogram b̂_k[0,ρ]. Values are cumulative
        counts of (observed events + expected events for censored) normalized
        by total F̂_k(∞). A well-calibrated model has b̂_k[0,ρ] ≈ ρ.

    See Also
    --------
    d_cr_calibration_per_event : Integrated calibration score per event.
    d_cr_calibration : Overall aggregated calibration score.
    aj_calibration : Aalen-Johansen calibration for competing risks.

    References
    ----------
    .. [Alberge2026] J. Alberge, T. Haugomat, G. Varoquaux, J. Abecassis,
        "On the calibration of survival models with competing risks",
        AISTATS 2026.
        <https://arxiv.org/pdf/2602.00194>
    """

    events, durations = check_y_survival(y_conf)
    fk = np.asarray(fk)
    fk_infty = np.asarray(fk_infty)
    s_t = np.asarray(s_t)

    # If times is provided, interpolate predictions to each individual's observed time
    if times is not None:
        times = np.asarray(times)
        durations_arr = np.asarray(durations)

        # Interpolate fk if 2D grid, else use as-is
        if fk.ndim == 2:
            fk = np.array(
                [
                    np.interp(durations_arr[i], times, fk[i, :])
                    for i in range(len(durations_arr))
                ]
            )

        # Interpolate fk_infty if 2D grid, else use as-is
        if fk_infty.ndim == 2:
            fk_infty = np.array(
                [
                    np.interp(durations_arr[i], times, fk_infty[i, :])
                    for i in range(len(durations_arr))
                ]
            )

        # Interpolate s_t if 2D grid, else use as-is
        if s_t.ndim == 2:
            s_t = np.array(
                [
                    np.interp(durations_arr[i], times, s_t[i, :])
                    for i in range(len(durations_arr))
                ]
            )

    bucket_edges = np.linspace(0, 1, n_buckets + 1)

    if event_of_interest == "any":
        event_mask = events > 0
    else:
        event_mask = events == event_of_interest

    fk_events = fk[event_mask]
    fk_infty_events = fk_infty[event_mask]
    fk_infty_all = fk_infty

    # Bin events by their normalized cumulative incidence
    normalized_inc = fk_events / (fk_infty_events + epsilon)
    event_bucket_indices = np.digitize(normalized_inc, bucket_edges, right=True)
    event_bucket_indices = np.clip(event_bucket_indices, 1, n_buckets)

    # Count events per bucket
    event_counts = np.zeros(n_buckets)
    unique_buckets, counts = np.unique(event_bucket_indices, return_counts=True)
    event_counts[unique_buckets - 1] = counts

    # Handle censored observations (delta_i = 0)
    censored_mask = events == 0
    if censored_mask.sum() == 0:
        # No censored data: normalize and return cumsum
        calibration = event_counts / fk_infty_all.sum()
        return np.cumsum(calibration)

    fk_censored = fk[censored_mask]
    fk_infty_censored = fk_infty[censored_mask]
    s_censored = s_t[censored_mask]

    normalized_inc_censored = fk_censored / (fk_infty_censored + epsilon)
    censored_contributions = np.zeros(n_buckets)
    bucket_width = 1.0 / n_buckets

    for bucket_idx in range(n_buckets):
        lower_edge = bucket_edges[bucket_idx]
        upper_edge = bucket_edges[bucket_idx + 1]

        # For censored obs with normalized_inc <= lower_edge: contribute
        # bucket_width * F_k(∞) / S(t) [approximation for low-risk censored]
        below_bucket = normalized_inc_censored <= lower_edge
        if below_bucket.any():
            censored_contributions[bucket_idx] += (
                (
                    bucket_width
                    * fk_infty_censored[below_bucket]
                    / (s_censored[below_bucket] + epsilon)
                )
            ).sum()

        # For censored obs in bucket: contribute (upper_edge * F_k(∞) - F_k(t)) / S(t)
        in_bucket = (normalized_inc_censored > lower_edge) & (
            normalized_inc_censored <= upper_edge
        )
        if in_bucket.any():
            censored_contributions[bucket_idx] += (
                (upper_edge * fk_infty_censored[in_bucket] - fk_censored[in_bucket])
                / (s_censored[in_bucket] + epsilon)
            ).sum()

    calibration = (event_counts + censored_contributions) / fk_infty_all.sum()
    return np.cumsum(calibration)


def d_cr_calibration_per_event(
    fk,
    fk_infty,
    s_t,
    y_conf,
    event_of_interest=None,
    alpha=2,
    epsilon=1e-3,
    n_buckets=100,
    times=None,
):
    r"""DCR-calibration score per event, integrated over risk buckets.

    Integrates the calibration deviation over the risk spectrum:

    .. math::

        \text{DCR-Cal}_k^\alpha = \frac{1}{\alpha}
        \int_0^1 |\hat{b}_k[0,\rho] - \rho|^\alpha \, d\rho

    where :math:`\hat{b}_k[0,\rho]` is the cumulative calibration value
    at bucket ρ, computed by :func:`d_calibration`.

    A score of zero indicates perfect calibration. A well-calibrated model
    satisfies :math:`\hat{b}_k[0,\rho] \approx \rho` across all buckets.

    Parameters
    ----------
    fk : array-like
        If times is None: shape (n_samples,), CIF for event k at observed times.
        If times is not None: shape (n_samples, n_times), CIF on time grid.

    fk_infty : array-like
        If times is None: shape (n_samples,), marginal event probability.
        If times is not None: shape (n_samples, n_times), on time grid.

    s_t : array-like
        If times is None: shape (n_samples,), survival at observed times.
        If times is not None: shape (n_samples, n_times), on time grid.

    y_conf : array-like of shape (n_samples, 2)
        Survival outcomes with "event" and "duration" columns.

    event_of_interest : int or None, default=None
        If provided, return only the score for that event.
        If ``None``, return a dict with one score per event.

    alpha : float, default=2
        Exponent applied to |b̂_k[0,ρ] - ρ| before integration.
        ``alpha=2`` gives squared L2 calibration; ``alpha=1`` gives L1.

    epsilon : float, default=1e-3
        Small constant for numerical stability.

    n_buckets : int, default=100
        Number of calibration buckets.

    times : array-like or None, default=None
        Time grid at which predictions are evaluated (optional).
        If provided, predictions will be interpolated to each individual's
        observed time. If None, fk, fk_infty, s_t are assumed to be
        exact values at observed times.
        Number of calibration buckets.

    Returns
    -------
    scores : dict of {int: float} or float
        Integrated calibration score per event. Returns a single float
        when ``event_of_interest`` is set.

    See Also
    --------
    d_calibration : Per-bucket calibration values.
    d_cr_calibration : Overall aggregated score across events.
    d_cr_calibration_ks_test : KS test for calibration.

    References
    ----------
    .. [Alberge2026] J. Alberge, T. Haugomat, G. Varoquaux, J. Abecassis,
        "On the calibration of survival models with competing risks",
        AISTATS 2026.
        <https://arxiv.org/pdf/2602.00194>
    """
    events, _ = check_y_survival(y_conf)
    event_ids = np.array(sorted(set([0]) | set(events)))

    scores = {}
    bucket_edges = np.linspace(0, 1, n_buckets + 1)

    for event_id in event_ids:
        if event_id == 0:
            continue

        b_hat = d_calibration(
            fk,
            fk_infty,
            s_t,
            y_conf,
            event_of_interest=event_id,
            epsilon=epsilon,
            n_buckets=n_buckets,
            times=times,
        )

        # Compute ∫ |b̂_k[0,ρ] - ρ|^α dρ using trapezoidal rule
        rho_values = bucket_edges[1:]  # bucket upper edges
        deviations = np.abs(b_hat - rho_values) ** alpha
        integral = np.trapezoid(deviations, rho_values)
        score = integral / alpha

        scores[event_id] = float(score)

    if event_of_interest is not None:
        return scores[event_of_interest]
    return scores


def d_cr_calibration(
    fk,
    fk_infty,
    s_t,
    y_conf,
    alpha=2,
    reduction="mean",
    epsilon=1e-3,
    n_buckets=100,
    times=None,
):
    r"""Overall DCR-calibration score aggregated across all events.

    Computes per-event DCR-calibration scores via
    :func:`d_cr_calibration_per_event`, then reduces them to a single number:

    .. math::

        \text{DCR-Cal} = \frac{1}{K} \sum_{k=1}^{K} \text{DCR-Cal}_k^\alpha
        \quad \text{(reduction='mean')}

    or the sum (resp. max) when ``reduction='sum'`` (resp. ``reduction='max'``).

    Each per-event score integrates :math:`|\hat{b}_k[0,\rho] - \rho|^\alpha`
    over the risk spectrum [0,1].

    Parameters
    ----------
    fk : array-like
        If times is None: shape (n_samples,), CIF at observed times.
        If times is not None: shape (n_samples, n_times), CIF on time grid.

    fk_infty : array-like
        If times is None: shape (n_samples,), marginal event probability.
        If times is not None: shape (n_samples, n_times), on time grid.

    s_t : array-like
        If times is None: shape (n_samples,), survival at observed times.
        If times is not None: shape (n_samples, n_times), on time grid.

    y_conf : array-like of shape (n_samples, 2)
        Survival outcomes with "event" and "duration" columns.

    alpha : float, default=2
        Exponent for calibration deviation.

    reduction : {"mean", "sum", "max"}, default="mean"
        How to aggregate per-event scores into a single value.

    epsilon : float, default=1e-3
        Small constant for numerical stability.

    n_buckets : int, default=100
        Number of calibration buckets.

    times : array-like or None, default=None
        Time grid at which predictions are evaluated (optional).
        If provided, predictions will be interpolated to each individual's
        observed time. If None, fk, fk_infty, s_t are assumed to be
        exact values at observed times.

    Returns
    -------
    score : float
        Aggregated DCR-calibration score.

    See Also
    --------
    d_cr_calibration_per_event : Per-event scores before aggregation.
    d_calibration : Per-bucket calibration values.
    d_cr_calibration_ks_test : KS test for calibration significance.

    References
    ----------
    .. [Alberge2026] J. Alberge, T. Haugomat, G. Varoquaux, J. Abecassis,
        "On the calibration of survival models with competing risks",
        AISTATS 2026.
        <https://arxiv.org/pdf/2602.00194>
    """
    if reduction not in ("mean", "sum", "max"):
        raise ValueError(
            f"reduction must be 'max', 'mean' or 'sum', got {reduction!r}."
        )

    scores = d_cr_calibration_per_event(
        fk,
        fk_infty,
        s_t,
        y_conf,
        alpha=alpha,
        epsilon=epsilon,
        n_buckets=n_buckets,
        times=times,
    )
    values = np.array(list(scores.values()))

    if reduction == "mean":
        return float(np.mean(values))
    elif reduction == "max":
        return float(np.max(values))
    return float(np.sum(values))


def d_cr_calibration_ks_test(
    fk,
    fk_infty,
    s_t,
    y_conf,
    event_of_interest=None,
    n_buckets=100,
    epsilon=1e-3,
    times=None,
):
    r"""Kolmogorov-Smirnov test for DCR-calibration.

    Tests whether the empirical calibration curve b̂_k[0,ρ] is
    significantly different from the identity line ρ (perfect calibration)
    using the KS test statistic:

    .. math::

        D_k = \max_\rho |\hat{b}_k[0,\rho] - \rho|

    Computes p-values under the null hypothesis of perfect calibration.
    A high p-value (e.g., > 0.05) indicates the model is well-calibrated.

    Parameters
    ----------
    fk : array-like
        If times is None: shape (n_samples,), CIF at observed times.
        If times is not None: shape (n_samples, n_times), CIF on time grid.

    fk_infty : array-like
        If times is None: shape (n_samples,), marginal event probability.
        If times is not None: shape (n_samples, n_times), on time grid.

    s_t : array-like
        If times is None: shape (n_samples,), survival at observed times.
        If times is not None: shape (n_samples, n_times), on time grid.

    y_conf : array-like of shape (n_samples, 2)
        Survival outcomes with "event" and "duration" columns.

    event_of_interest : int or None, default=None
        If provided, return only the test for that event.
        If ``None``, return results for all events.

    n_buckets : int, default=100
        Number of calibration buckets.

    epsilon : float, default=1e-3
        Small constant for numerical stability.

    times : array-like or None, default=None
        Time grid at which predictions are evaluated (optional).
        If provided, predictions will be interpolated to each individual's
        observed time. If None, fk, fk_infty, s_t are assumed to be
        exact values at observed times.

    Returns
    -------
    results : dict of {int: dict} or dict
        For each event, contains:
        - "statistic": KS test statistic (maximum absolute deviation)
        - "pvalue": p-value under null hypothesis of perfect calibration
        When ``event_of_interest`` is set, returns only the results dict.

    Notes
    -----
    The p-value is computed using the Kolmogorov distribution. A low p-value
    suggests the model's predictions deviate significantly from perfect
    calibration at some risk level.

    See Also
    --------
    d_calibration : Per-bucket calibration values.
    d_cr_calibration_per_event : Integrated calibration score.
    d_cr_calibration : Overall aggregated calibration score.

    References
    ----------
    .. [Alberge2026] J. Alberge, T. Haugomat, G. Varoquaux, J. Abecassis,
        "On the calibration of survival models with competing risks",
        AISTATS 2026.
        <https://arxiv.org/pdf/2602.00194>
    """
    events, _ = check_y_survival(y_conf)
    event_ids = np.array(sorted(set([0]) | set(events)))

    bucket_edges = np.linspace(0, 1, n_buckets + 1)
    rho_values = bucket_edges[1:]  # bucket upper edges

    results = {}

    for event_id in event_ids:
        if event_id == 0:
            continue

        b_hat = d_calibration(
            fk,
            fk_infty,
            s_t,
            y_conf,
            event_of_interest=event_id,
            epsilon=epsilon,
            n_buckets=n_buckets,
            times=times,
        )

        # KS statistic: maximum absolute deviation from identity
        ks_statistic = float(np.max(np.abs(b_hat - rho_values)))

        # Compute p-value using KS distribution
        # For large n, use approximation from Kolmogorov distribution
        # pvalue ≈ 2 * exp(-2 * n * D^2) where n is effective sample size
        n_effective = len(events[events == event_id])
        if n_effective > 0:
            pvalue = 2 * np.exp(-2 * n_effective * ks_statistic**2)
            pvalue = np.clip(pvalue, 0, 1)
        else:
            pvalue = 1.0

        results[event_id] = {
            "statistic": ks_statistic,
            "pvalue": pvalue,
        }

    if event_of_interest is not None:
        return results[event_of_interest]
    return results
