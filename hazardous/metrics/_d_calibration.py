import numpy as np

from ..utils import check_event_of_interest, check_y_survival


def d_calibration(
    fk,
    fk_infty,
    s_t,
    y_conf,
    event_of_interest="any",
    epsilon=1e-3,
    n_buckets=100,
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
    fk : ndarray, shape (n_samples,)
        Predicted cumulative incidence function for event k evaluated
        at each individual's observed time: F̂_k(tᵢ|xᵢ).
        dtype: float

    fk_infty : ndarray, shape (n_samples,)
        Marginal event probability at infinite time: F̂_k(∞|xᵢ).
        dtype: float

    s_t : ndarray, shape (n_samples,)
        Predicted survival probability at observed time: Ŝ(tᵢ|xᵢ).
        dtype: float

    y_conf : dict or structured array, shape (n_samples,)
        Survival outcomes with two fields:
            "event": ndarray of shape (n_samples,), dtype int
                0 for censored, positive integers for event type.
            "duration": ndarray of shape (n_samples,), dtype float
                Observed time (either event or censoring time).

    event_of_interest : {"any", int}, default="any"
        Which event to compute calibration for.
        - "any": any event occurred (δᵢ > 0)
        - int: specific event type (δᵢ = event_of_interest)

    epsilon : float, default=1e-3
        Small constant added to denominators for numerical stability.

    n_buckets : int, default=100
        Number of equal-width buckets for the calibration histogram.
        Creates buckets at edges [0, 1/n_buckets, 2/n_buckets, ..., 1].

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
    check_event_of_interest(event_of_interest)
    fk = np.asarray(fk)
    fk_infty = np.asarray(fk_infty)
    s_t = np.asarray(s_t)

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
            delta = upper_edge * fk_infty_censored[in_bucket] - fk_censored[in_bucket]
            censored_contributions[bucket_idx] += (
                np.maximum(delta, 0) / (s_censored[in_bucket] + epsilon)
            ).sum()

    calibration = (event_counts + censored_contributions) / fk_infty_all.sum()
    return np.cumsum(calibration)


def d_cr_calibration_per_event(
    y_conf,
    event_of_interest=None,
    alpha=2,
    epsilon=1e-3,
    n_buckets=100,
    exact=False,
    fk_t=None,
    fk_infty=None,
    s_t=None,
    y_conf_pred=None,
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
    y_conf : dict or structured array, shape (n_samples,)
        Survival outcomes with two fields:
            "event": ndarray of shape (n_samples,), dtype int
                0 for censored, positive integers for event type.
            "duration": ndarray of shape (n_samples,), dtype float
                Observed time.

    event_of_interest : int or None, default=None
        Which event(s) to compute calibration for.

        If None: Compute for all events found in y_conf.
            Returns: dict of {event_id: score}.

        If int (e.g., 1, 2, 3): Compute only for that event.
            Returns: float (single score).

    alpha : float, default=2
        Exponent applied to |b̂_k[0,ρ] - ρ| before integration.
        ``alpha=2`` gives squared L2 calibration; ``alpha=1`` gives L1.

    epsilon : float, default=1e-3
        Small constant for numerical stability.

    n_buckets : int, default=100
        Number of calibration buckets.
        Creates bucket edges at [0, 1/n, 2/n, ..., 1].

    exact : bool, default=False
        Whether predictions are already evaluated at observed times.

        If False (default): Use interpolation approach
            Requires: y_conf_pred and times parameters.
            Automatically extracts and interpolates predictions.

        If True: Use exact approach
            Requires: fk_t, fk_infty, s_t parameters.
            User has pre-evaluated predictions at observed times.

    fk_t : ndarray or None, default=None
        CIF for event(s) at observed times. Required if exact=True.

        Single event:
            shape: (n_samples,)
            dtype: float
            Used when event_of_interest is specified.

    fk_infty : ndarray or None, default=None
        Marginal event probability. Required if exact=True.

        shape: (n_samples,)
        dtype: float
        Event probability at infinity.

    s_t : ndarray or None, default=None
        Survival probability at observed times. Required if exact=True.

        shape: (n_samples,)
        dtype: float

    y_conf_pred : ndarray or None, default=None
        Model predictions for all events at all times.
        Required if exact=False (default).

        shape: (n_samples, n_events+1, n_times)
        dtype: float
        y_conf_pred[:, 0, :] = survival probabilities at all times
        y_conf_pred[:, i, :] = CIF for event i at all times

    times : ndarray or None, default=None
        Time grid at which predictions are evaluated.
        Required if exact=False (default).

        shape: (n_times,)
        dtype: float
        Time points where y_conf_pred is computed.

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
    events, durations = check_y_survival(y_conf)
    event_ids = np.array(sorted(set([0]) | set(events)))

    # Extract predictions based on exact flag
    if exact:
        # User provided exact predictions at observed times
        if fk_t is None or fk_infty is None or s_t is None:
            raise ValueError("If exact=True, must provide fk_t, fk_infty, and s_t")
        fk_t = np.asarray(fk_t)
        fk_infty = np.asarray(fk_infty)
        s_t = np.asarray(s_t)

        # Determine which events to compute for
        if event_of_interest is not None:
            events_to_compute = [event_of_interest]
        else:
            events_to_compute = [e for e in event_ids if e > 0]
    else:
        # Extract from time-grid predictions via interpolation
        if y_conf_pred is None or times is None:
            raise ValueError(
                "If exact=False (default), must provide y_conf_pred and times"
            )

        y_conf_pred = np.asarray(y_conf_pred)
        times = np.asarray(times)
        durations = np.asarray(durations)

        if y_conf_pred.ndim != 3:
            raise ValueError(
                "y_conf_pred must be 3D: (n_samples, n_events+1, n_times), "
                f"got shape {y_conf_pred.shape}"
            )

        n_samples, n_events_plus_1, n_times = y_conf_pred.shape
        n_events = n_events_plus_1 - 1

        # Interpolate to observed times for each sample and event
        fk_t = np.zeros((n_samples, n_events))
        fk_infty = np.zeros((n_samples, n_events))
        s_t = np.zeros(n_samples)

        for i in range(n_samples):
            # Survival at observed time
            s_t[i] = np.interp(durations[i], times, y_conf_pred[i, 0, :])

            # Each event's CIF at observed time and at infinity
            for j in range(n_events):
                fk_t[i, j] = np.interp(durations[i], times, y_conf_pred[i, j + 1, :])
                fk_infty[i, j] = y_conf_pred[i, j + 1, -1]

        # Determine which events to compute for
        if event_of_interest is not None:
            events_to_compute = [event_of_interest]
        else:
            events_to_compute = [e for e in event_ids if e > 0]

    scores = {}
    bucket_edges = np.linspace(0, 1, n_buckets + 1)

    for event_id in events_to_compute:
        # Extract event-specific predictions
        if exact:
            # fk_t is 1D, use directly
            fk_t_event = fk_t
            fk_infty_event = fk_infty
        else:
            # fk_t is 2D (n_samples, n_events), extract column
            fk_t_event = fk_t[:, event_id - 1]
            fk_infty_event = fk_infty[:, event_id - 1]

        b_hat = d_calibration(
            fk_t_event,
            fk_infty_event,
            s_t,
            y_conf,
            event_of_interest=event_id,
            epsilon=epsilon,
            n_buckets=n_buckets,
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
    y_conf,
    alpha=2,
    reduction="mean",
    epsilon=1e-3,
    n_buckets=100,
    exact=False,
    fk_t=None,
    fk_infty=None,
    s_t=None,
    y_conf_pred=None,
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
    y_conf : dict or structured array, shape (n_samples,)
        Survival outcomes with two fields:
            "event": ndarray of shape (n_samples,), dtype int
                0 for censored, positive integers for each event type.
            "duration": ndarray of shape (n_samples,), dtype float
                Observed time.

    alpha : float, default=2
        Exponent for calibration deviation.
        ``alpha=2`` gives squared L2 calibration; ``alpha=1`` gives L1.

    reduction : {"mean", "sum", "max"}, default="mean"
        How to aggregate per-event scores into a single value:
        - "mean": (1/K) * Σ scores_k
        - "sum": Σ scores_k
        - "max": max(scores_k)

    epsilon : float, default=1e-3
        Small constant for numerical stability.

    n_buckets : int, default=100
        Number of calibration buckets.
        Creates bucket edges at [0, 1/n, 2/n, ..., 1].

    exact : bool, default=False
        Whether predictions are already evaluated at observed times.

        If False (default): Use interpolation approach
            Requires: y_conf_pred and times parameters.
            Automatically extracts and interpolates predictions.

        If True: Use exact approach
            Requires: fk_t, fk_infty, s_t parameters.
            User has pre-evaluated predictions at observed times.

    fk_t : ndarray or None, default=None
        CIF for event(s) at observed times. Required if exact=True.
        shape: (n_samples,), dtype: float

    fk_infty : ndarray or None, default=None
        Marginal event probability. Required if exact=True.
        shape: (n_samples,), dtype: float

    s_t : ndarray or None, default=None
        Survival probability at observed times. Required if exact=True.
        shape: (n_samples,), dtype: float

    y_conf_pred : ndarray of shape (n_samples, n_events+1, n_times)
        or None, default=None
        Model predictions for all events at all times.
        Required if exact=False (default).

    times : ndarray of shape (n_times,) or None, default=None
        Time grid at which predictions are evaluated.
        Required if exact=False (default).

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
        y_conf,
        alpha=alpha,
        epsilon=epsilon,
        n_buckets=n_buckets,
        exact=exact,
        fk_t=fk_t,
        fk_infty=fk_infty,
        s_t=s_t,
        y_conf_pred=y_conf_pred,
        times=times,
    )
    values = np.array(list(scores.values()))

    if reduction == "mean":
        return float(np.mean(values))
    elif reduction == "max":
        return float(np.max(values))
    return float(np.sum(values))


def d_cr_calibration_ks_test(
    y_conf,
    event_of_interest=None,
    n_buckets=100,
    epsilon=1e-3,
    exact=False,
    fk_t=None,
    fk_infty=None,
    s_t=None,
    y_conf_pred=None,
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
    fk : ndarray
        Predicted cumulative incidence function for event(s).

        Single event (1D):
            shape: (n_samples,)
            dtype: float
            CIF for a single event at observed times.
            Requires ``event_of_interest`` to be specified.

        Multiple events (2D, competing risks):
            shape: (n_samples, n_events)
            dtype: float
            fk[:, i-1] is the CIF for event i.
            Returns tests for all events in y_conf.

        Grid-based interpolation (2D):
            shape: (n_samples, n_times)
            dtype: float
            When ``times`` is not None, CIF on time grid.

    fk_infty : ndarray
        Marginal event probabilities (F̂_k(∞|xᵢ)).

        Single event (1D):
            shape: (n_samples,)
            dtype: float

        Multiple events (2D):
            shape: (n_samples, n_events)
            dtype: float
            fk_infty[:, i-1] is marginal for event i.

        Grid-based (2D):
            shape: (n_samples, n_times)
            dtype: float

    s_t : ndarray
        Predicted survival function (Ŝ(tᵢ|xᵢ)).

        Exact times:
            shape: (n_samples,)
            dtype: float

        Grid-based interpolation:
            shape: (n_samples, n_times)
            dtype: float
        Same for all events (not event-specific).

    y_conf : dict or structured array, shape (n_samples,)
        Survival outcomes with two fields:
            "event": ndarray of shape (n_samples,), dtype int
                0 for censored, positive integers for event type.
            "duration": ndarray of shape (n_samples,), dtype float
                Observed time.

    event_of_interest : int or None, default=None
        Which event(s) to test.

        If None: Test all events found in y_conf.
            fk must be 2D (n_samples, n_events).
            Returns: dict of {event_id: test_results}.

        If int (e.g., 1, 2, 3): Test only that event.
            fk can be 1D or 2D.
            Returns: dict with single event's results.

    n_buckets : int, default=100
        Number of calibration buckets.
        Creates bucket edges at [0, 1/n, 2/n, ..., 1].

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
    events, durations = check_y_survival(y_conf)
    event_ids = np.array(sorted(set([0]) | set(events)))

    # Extract predictions based on exact flag
    if exact:
        # User provided exact predictions at observed times
        if fk_t is None or fk_infty is None or s_t is None:
            raise ValueError("If exact=True, must provide fk_t, fk_infty, and s_t")
        fk_t = np.asarray(fk_t)
        fk_infty = np.asarray(fk_infty)
        s_t = np.asarray(s_t)

        # Determine which events to test
        if event_of_interest is not None:
            events_to_test = [event_of_interest]
        else:
            events_to_test = [e for e in event_ids if e > 0]
    else:
        # Extract from time-grid predictions via interpolation
        if y_conf_pred is None or times is None:
            raise ValueError(
                "If exact=False (default), must provide y_conf_pred and times"
            )

        y_conf_pred = np.asarray(y_conf_pred)
        times = np.asarray(times)
        durations = np.asarray(durations)

        if y_conf_pred.ndim != 3:
            raise ValueError(
                "y_conf_pred must be 3D: (n_samples, n_events+1, n_times), "
                f"got shape {y_conf_pred.shape}"
            )

        n_samples, n_events_plus_1, n_times = y_conf_pred.shape
        n_events = n_events_plus_1 - 1

        # Interpolate to observed times
        fk_t = np.zeros((n_samples, n_events))
        fk_infty = np.zeros((n_samples, n_events))
        s_t = np.zeros(n_samples)

        for i in range(n_samples):
            s_t[i] = np.interp(durations[i], times, y_conf_pred[i, 0, :])
            for j in range(n_events):
                fk_t[i, j] = np.interp(durations[i], times, y_conf_pred[i, j + 1, :])
                fk_infty[i, j] = y_conf_pred[i, j + 1, -1]

        # Determine which events to test
        if event_of_interest is not None:
            events_to_test = [event_of_interest]
        else:
            events_to_test = [e for e in event_ids if e > 0]

    bucket_edges = np.linspace(0, 1, n_buckets + 1)
    rho_values = bucket_edges[1:]  # bucket upper edges

    results = {}

    for event_id in events_to_test:
        # Extract event-specific predictions
        if exact:
            fk_t_event = fk_t
            fk_infty_event = fk_infty
        else:
            fk_t_event = fk_t[:, event_id - 1]
            fk_infty_event = fk_infty[:, event_id - 1]

        b_hat = d_calibration(
            fk_t_event,
            fk_infty_event,
            s_t,
            y_conf,
            event_of_interest=event_id,
            epsilon=epsilon,
            n_buckets=n_buckets,
        )

        # KS statistic: maximum absolute deviation from identity
        ks_statistic = float(np.max(np.abs(b_hat - rho_values)))

        # Compute p-value using KS distribution
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
