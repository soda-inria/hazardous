import numpy as np

from ..utils import check_y_survival


def accuracy_in_time(y_test, y_pred, time_grid, quantiles=None, taus=None):
    r"""Accuracy in time for prognostic models using competing risks.

    .. math::

        \mathrm{acc}(\zeta) = \frac{1}{n_{nc}} \sum_{i=1}^n ~ I\{\hat{y}_i=y_{i,\zeta}\}
        \overline{I\{\delta_i = 0 \cap t_i \leq \zeta \}}

    where:

    - :math:`I` is the indicator function.
    - :math:`\zeta` is a fixed time horizon.
    - :math:`n_{nc}` is the number of uncensored individuals at :math:`\zeta`.
    - :math:`\delta_i` is the event experienced by the individual :math:`i` at
      :math:`t_i`.
    - :math:`\hat{y} = \text{arg}\max\limits_{k \in [0, K]} \hat{F}_k(\zeta|X=x_i)`
      where :math:`\hat{F}_0(\zeta|X=x_i) \triangleq \hat{S}(\zeta|X=x_i)`.

      :math:`\hat{y}` is the most probable predicted event for individual :math:`i`
      at :math:`\zeta`.
    - :math:`y_{i,\zeta} = \delta_i ~ I\{t_i \leq \zeta \}` is the observed event
      for individual :math:`i` at :math:`\zeta`.

    The accuracy in time is a metric introduced in [Alberge2024]_ which evaluates
    whether observed events are predicted as the most likely at given times.
    This metric measures if the highest predicted event (one of the event of interest
    or the survival one) corresponds to the one observed at :math:`\zeta` for each
    patient.

    We remove individuals that were censored at times :math:`t \leq \zeta`, so the
    accuracy in time essentially represents the accuracy of the estimator on
    observed events up to :math:`\zeta`.

    In the beginning, every model's accuracy in time will be high because it will
    predict that the patients have survived, which will be true in most cases. This
    metric's discriminative power will be for advanced times when the model has to
    select which event will happen for a given patient.

    The C-index depends on other individual in the cohort, while the accuracy-in-time
    for an individual does not. Conceptually, the C-index can help clinicians to
    priorize treatment allocation by ranking individuals by risk of a given event of
    interest. The accuracy in time, however, answers a different question: "`what is
    the most likely event that this individual will experience at some fixed time
    horizon?`". Therefore, the accuracy in time helps clinicians choose the right
    treatment by priorizing the risk for a given individual.

    Parameters
    ----------
    y_test : array, dictionnary or dataframe of shape (n_samples, 2)
        The test target, consisting in the 'event' and 'duration' columns

    y_pred : array of shape (n_samples_test, n_events, n_time_grid)
        Cumulative incidence for all competing events, at the time points
        from the input time_grid.

    time_grid : array of shape (n_time_grid,)
        Time points used to predict the cumulative incidence.

    quantiles : array or list of shape (n_quantiles,), default=None
        The quantiles of ``time_grid`` used to define the fixed horizons at which
        to compute the accuracy in time. The values of the quantiles of ``time_grid``
        are equivalent to ``taus``, therefore ``quantiles`` can't be set if ``taus``
        is set. If neither ``taus`` nor ``quantiles`` are set, we set quantiles
        as a uniform grid of 8 quantiles, from 0 to 1.

    taus : array or list of shape (n_taus), default=None
        The fixed time horizons to compute the accuracy in time. Can't be set if
        ``quantiles`` is set.

    Returns
    -------
    acc_in_time : array of shape (n_quantiles or n_taus)
        The accuracy in time computed at the fixed horizons ``taus``.

    taus : array of shape (n_quantiles or n_taus)
        The fixed time horizons effectively used to compute the accuracy in time.

    References
    ----------
    .. [Alberge2024] J. Alberge, V. Maladiere,  O. Grisel, J. Abécassis, G. Varoquaux,
        "Survival Models: Proper Scoring Rule and Stochastic Optimization
        with Competing Risks", 2024
    """
    event_true, _ = check_y_survival(y_test)

    if y_pred.ndim != 3:
        raise ValueError(
            "'y_pred' must be a 3D array with shape (n_samples, n_events, n_times), got"
            f" shape {y_pred.shape}."
        )
    if y_pred.shape[0] != event_true.shape[0]:
        raise ValueError(
            "'y_true' and 'y_pred' must have the same number of samples, "
            f"got {event_true.shape[0]} and {y_pred.shape[0]} respectively."
        )
    time_grid = np.atleast_1d(time_grid)
    if y_pred.shape[2] != time_grid.shape[0]:
        raise ValueError(
            f"'time_grid' length ({time_grid.shape[0]}) "
            f"must be equal to y_pred.shape[2] ({y_pred.shape[2]})."
        )

    if quantiles is not None:
        if taus is not None:
            raise ValueError("'quantiles' and 'taus' can't be set at the same time.")

        quantiles = np.atleast_1d(quantiles)
        if any(quantiles < 0) or any(quantiles > 1):
            raise ValueError(f"quantiles must be in [0, 1], got {quantiles}.")
        taus = np.quantile(time_grid, quantiles)

    elif quantiles is None and taus is None:
        n_quantiles = min(time_grid.shape[0], 8)
        quantiles = np.linspace(0, 1, n_quantiles)
        taus = np.quantile(time_grid, quantiles)

    acc_in_time = []

    for tau in taus:
        mask_past_censored = (y_test["event"] == 0) & (y_test["duration"] <= tau)

        tau_idx = np.searchsorted(time_grid, tau)

        # If tau is beyond the time_grid, we extrapolate its accuracy as
        # the accuracy at max(time_grid).
        if tau_idx == time_grid.shape[0]:
            tau_idx = -1

        y_pred_at_t = y_pred[:, :, tau_idx]
        y_pred_class = y_pred_at_t[~mask_past_censored, :].argmax(axis=1)

        y_test_class = y_test["event"] * (y_test["duration"] <= tau)
        y_test_class = y_test_class.loc[~mask_past_censored].values

        acc_in_time.append((y_test_class == y_pred_class).mean())

    return np.array(acc_in_time), np.asarray(taus)
