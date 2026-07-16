import numpy as np
import pandas as pd

from ..utils import check_y_survival


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
    fk : array-like of shape (n_samples,)
        Predicted cumulative incidence function for event k evaluated
        at the observed time: F̂_k(tᵢ|xᵢ).

    fk_infty : array-like of shape (n_samples,)
        Predicted incidence at infinite time (marginal event probability):
        F̂_k(∞|xᵢ).

    s_t : array-like of shape (n_samples,)
        Predicted survival function evaluated at the observed time:
        Ŝ(tᵢ|xᵢ).

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

    Returns
    -------
    calibration : DataFrame of shape (n_buckets,)
        Cumulative calibration histogram b̂_k[0,ρ]. Index is bucket number
        (1 to n_buckets). Values are cumulative counts of (observed events +
        expected events for censored) normalized by total F̂_k(∞).
        A well-calibrated model has b̂_k[0,ρ] ≈ ρ.

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

    Examples
    --------
    >>> import numpy as np
    >>> from hazardous.metrics import d_calibration
    >>> n_samples = 200
    >>> y_conf = {
    ...     "event": np.array([0, 1, 2] * (n_samples // 3)),
    ...     "duration": np.random.exponential(10, n_samples)
    ... }
    >>> # Predictions for event 1 evaluated at observed time for each individual
    >>> fk = np.random.uniform(0, 0.5, n_samples)  # F̂_1(tᵢ|xᵢ)
    >>> fk_infty = np.random.uniform(0.4, 0.9, n_samples)  # F̂_1(∞|xᵢ)
    >>> s_t = np.random.uniform(0.3, 1.0, n_samples)  # Ŝ(tᵢ|xᵢ)
    >>> calib = d_calibration(fk, fk_infty, s_t, y_conf, event_of_interest=1)
    >>> print(calib.head())
    """

    events, durations = check_y_survival(y_conf)
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
    event_counts = pd.Series(0, index=range(1, n_buckets + 1))
    unique_buckets, counts = np.unique(event_bucket_indices, return_counts=True)
    event_counts.loc[unique_buckets] = counts

    # Handle censored observations (delta_i = 0)
    censored_mask = events == 0
    if censored_mask.sum() == 0:
        # No censored data: normalize and return cumsum
        calibration = event_counts / fk_infty_all.sum()
        return pd.DataFrame(calibration, columns=["calibration"]).cumsum()

    fk_censored = fk[censored_mask]
    fk_infty_censored = fk_infty[censored_mask]
    s_censored = s_t[censored_mask]

    df_censored = pd.DataFrame(
        {
            "normalized_inc": fk_censored / (fk_infty_censored + epsilon),
            "fk_infty": fk_infty_censored,
            "fk": fk_censored,
            "s": s_censored + epsilon,
        }
    )

    censored_contributions = pd.Series(0.0, index=range(1, n_buckets + 1))
    bucket_width = 1.0 / n_buckets

    for bucket_idx in range(1, n_buckets + 1):
        lower_edge = bucket_edges[bucket_idx - 1]
        upper_edge = bucket_edges[bucket_idx]

        # For censored obs with normalized_inc <= lower_edge: contribute
        # bucket_width * F_k(∞) / S(t) [approximation for low-risk censored]
        below_bucket = df_censored["normalized_inc"] <= lower_edge
        if below_bucket.sum() > 0:
            censored_contributions[bucket_idx] += (
                (
                    bucket_width
                    * df_censored.loc[below_bucket, "fk_infty"]
                    / df_censored.loc[below_bucket, "s"]
                )
            ).sum()

        # For censored obs in bucket: contribute (upper_edge * F_k(∞) - F_k(t)) / S(t)
        # where upper_edge is the upper bound of the current bucket
        in_bucket = (df_censored["normalized_inc"] > lower_edge) & (
            df_censored["normalized_inc"] <= upper_edge
        )
        if in_bucket.sum() > 0:
            censored_contributions[bucket_idx] += (
                (
                    upper_edge * df_censored.loc[in_bucket, "fk_infty"]
                    - df_censored.loc[in_bucket, "fk"]
                )
                / df_censored.loc[in_bucket, "s"]
            ).sum()

    calibration = (event_counts + censored_contributions) / fk_infty_all.sum()
    return pd.DataFrame(calibration, columns=["calibration"]).cumsum()


def d_cr_calibration_per_event(
    fk,
    fk_infty,
    s_t,
    y_conf,
    event_of_interest=None,
    alpha=2,
    epsilon=1e-3,
    n_buckets=100,
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
    fk : array-like of shape (n_samples,)
        Predicted cumulative incidence function for event k.

    fk_infty : array-like of shape (n_samples,)
        Predicted marginal event probability.

    s_t : array-like of shape (n_samples,)
        Predicted survival function at observed time.

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

        calib_df = d_calibration(
            fk,
            fk_infty,
            s_t,
            y_conf,
            event_of_interest=event_id,
            epsilon=epsilon,
            n_buckets=n_buckets,
        )
        b_hat = calib_df.values.flatten()

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
    fk : array-like of shape (n_samples,)
        Predicted cumulative incidence function.

    fk_infty : array-like of shape (n_samples,)
        Predicted marginal event probability.

    s_t : array-like of shape (n_samples,)
        Predicted survival function.

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
    fk : array-like of shape (n_samples,)
        Predicted cumulative incidence function.

    fk_infty : array-like of shape (n_samples,)
        Predicted marginal event probability.

    s_t : array-like of shape (n_samples,)
        Predicted survival function.

    y_conf : array-like of shape (n_samples, 2)
        Survival outcomes with "event" and "duration" columns.

    event_of_interest : int or None, default=None
        If provided, return only the test for that event.
        If ``None``, return results for all events.

    n_buckets : int, default=100
        Number of calibration buckets.

    epsilon : float, default=1e-3
        Small constant for numerical stability.

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

        calib_df = d_calibration(
            fk,
            fk_infty,
            s_t,
            y_conf,
            event_of_interest=event_id,
            epsilon=epsilon,
            n_buckets=n_buckets,
        )
        b_hat = calib_df.values.flatten()

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
