r"""AJ-Calibration: marginal calibration for competing risks models.

This module implements the AJ-Calibration metric, which generalises
KM-Calibration to competing risks by comparing the mean predicted cumulative
incidence functions (CIFs) against the Aalen-Johansen marginal estimator.

For each cause of event :math:`k \in \{1, \ldots, K\}` the score is:

.. math::

    \text{AJ-Cal}_k = \frac{1}{t_{\max}} \int_0^{t_{\max}}
    \left(\bar{F}_k(t) - \hat{F}^{AJ}_k(t)\right)^\alpha \, dt

where:

- :math:`\bar{F}_k(t) = \frac{1}{n} \sum_{i=1}^n
  \hat{F}_k(t \mid \mathbf{x}_i)` is the mean of the predicted CIFs across
  the calibration cohort.

- :math:`\hat{F}^{AJ}_k(t)` is the marginal Aalen-Johansen CIF for event
  :math:`k`, fitted on the same calibration cohort.

- :math:`\alpha \geq 1` controls sensitivity to large deviations (default
  :math:`\alpha = 2`, giving a squared L2 calibration score).

The survival calibration (event 0) is handled by the KM-Calibration metric
from :mod:`hazardous.metrics._km_calibration`.

References
----------
.. [Alberge2026] J. Alberge, T. Haugomat, G.Varoquaux,
    J. Abecassis,  "On the calibration of survival models with competing risks",
    arXiv:2602.00194, 2026.
    https://arxiv.org/pdf/2602.00194
"""

import numpy as np

from .._km_sampler import _AalenJohansenSampler
from ..metrics._km_calibration import KMCalibration
from ..utils import check_y_survival


class AJCalibration:
    r"""Marginal calibration of a competing risks model via Aalen-Johansen.

    Measures how closely the mean predicted cumulative incidence functions
    (CIFs) track the marginal Aalen-Johansen estimates over a given time
    grid. For each cause of event :math:`k` the calibration score is:

    .. math::

        \text{AJ-Cal}_k = \frac{1}{t_{\max}} \int_0^{t_{\max}}
        \left(\bar{F}_k(t) - \hat{F}^{AJ}_k(t)\right)^\alpha \, dt

    where :math:`\bar{F}_k(t)` is the mean predicted CIF across the
    calibration cohort and :math:`\hat{F}^{AJ}_k(t)` is the
    Aalen-Johansen CIF for event :math:`k`.

    The calibration for the survival probability (event 0, key ``0`` in
    the output) is computed via :class:`~hazardous.metrics.KMCalibration`.

    Parameters
    ----------
    alpha : int, default=2
        Exponent applied to the pointwise difference before integration.
        When ``alpha=2``, the score is an L2 (squared) calibration score.

    Attributes
    ----------
    aalen_johansen_sampler_ : _AalenJohansenSampler
        Fitted Aalen-Johansen sampler used to estimate the marginal CIFs.

    km_calibration_ : KMCalibration
        Fitted KM-Calibration object used to score the survival probability
        (event 0).

    event_ids_ : ndarray
        Sorted array of unique event identifiers observed during fitting,
        including 0 for censoring.

    See Also
    --------
    aj_calibration : Functional API for this class.
    KMCalibration : KM-based calibration for single-event survival.

    References
    ----------
    .. [Alberge2026] J. Alberge, T. Haugomat, G.Varoquaux,
        J. Abecassis,  "On the calibration of survival models with competing risks",
        arXiv:2602.00194, 2026.
        https://arxiv.org/pdf/2602.00194

    Examples
    --------
    >>> import numpy as np
    >>> from hazardous.metrics import AJCalibration
    >>> rng = np.random.default_rng(0)
    >>> n = 300
    >>> duration = rng.exponential(scale=10, size=n)
    >>> event = rng.choice([0, 1, 2], size=n)
    >>> y = {"event": event, "duration": duration}
    >>> times = np.linspace(0, 20, 30)
    >>> # Perfect calibration: predictions equal AJ estimates
    >>> from lifelines import AalenJohansenFitter
    >>> aj1 = AalenJohansenFitter().fit(
            durations=duration,
            event_observed=event,
            event_of_interest=1,
        )
    >>> aj2 = AalenJohansenFitter().fit(
            durations=duration,
            event_observed=event,
            event_of_interest=2,
        )
    >>> n_events = 2  # event ids 0, 1, 2
    >>> surv = np.tile(aj1.survival_func_, (n, 1))
    >>> cif1 = np.tile(aj1.incidence_func_, (n, 1))
    >>> cif2 = np.tile(aj2.incidence_func_, (n, 1))
    >>> inc_pred = np.stack([surv, cif1, cif2], axis=1)
    >>> cal = AJCalibration().fit(y)
    >>> scores = cal.score(times, inc_pred)
    >>> all(abs(v) < 1e-10 for v in scores.values())
    True
    """

    def __init__(self, alpha=2):
        self.alpha = alpha

    def fit(self, y_conf):
        """Fit the Aalen-Johansen estimator on the calibration cohort.

        Parameters
        ----------
        y_conf : array-like of shape (n_samples, 2)
            Survival outcomes of the calibration cohort, with columns
            ``"event"`` (0 for censoring, positive integers for each cause
            of event) and ``"duration"`` (observed time).

        Returns
        -------
        self : object
            Fitted estimator.
        """
        event, _ = check_y_survival(y_conf)
        self.event_ids_ = np.array(sorted(set([0]) | set(event)))

        self.aalen_johansen_sampler_ = _AalenJohansenSampler().fit(y_conf)
        self.km_calibration_ = KMCalibration(alpha=self.alpha).fit(y_conf)
        return self

    def score(self, times, inc_prob_at_conf):
        """Compute AJ-Calibration scores for all causes of event.

        Parameters
        ----------
        times : array-like of shape (n_times,)
            Time points at which the CIFs were predicted. Need not be
            sorted; the last axis of ``inc_prob_at_conf`` must correspond
            to the same ordering as ``times``.

        inc_prob_at_conf : array-like of shape (n_samples, n_events+1, n_times)
            Predicted incidence probabilities at ``times`` for the
            calibration cohort. The second axis indexes event identifiers in
            sorted order: axis 0 holds the survival probability (event 0),
            and axes 1, 2, … hold cause-specific CIFs.

        Returns
        -------
        scores : dict of {int: float}
            AJ-Calibration score for each event identifier. Key ``0``
            corresponds to the survival KM-Calibration score.
        """
        times, inc_prob_at_conf = self._sort_by_time(times, inc_prob_at_conf)
        t_max = times[-1]

        scores = {}
        scores[0] = self.km_calibration_.score(times, inc_prob_at_conf[:, 0, :])

        for event_id in self.event_ids_[1:]:
            inc_func_aj = self.aalen_johansen_sampler_.incidence_func_[event_id]
            inc_probs_aj = inc_func_aj(times)
            inc_probs_mean = inc_prob_at_conf[:, event_id, :].mean(axis=0)
            diff_at_t = inc_probs_mean - inc_probs_aj
            scores[event_id] = np.trapz(diff_at_t**self.alpha, times) / t_max

        return scores

    def difference_at_t(self, times, inc_prob_at_conf):
        """Compute the pointwise difference between mean predictions and AJ.

        Parameters
        ----------
        times : array-like of shape (n_times,)
            Time points at which the CIFs were predicted.

        inc_prob_at_conf : array-like of shape (n_samples, n_events+1, n_times)
            Predicted incidence probabilities at ``times`` for the
            calibration cohort.

        Returns
        -------
        differences : dict of {int: ndarray of shape (n_times,)}
            Pointwise difference :math:`\\bar{F}_k(t) - \\hat{F}^{AJ}_k(t)`
            for each event identifier, in ascending time order. Key ``0``
            holds the survival difference against the KM estimate.
        """
        times, inc_prob_at_conf = self._sort_by_time(times, inc_prob_at_conf)

        differences = {}
        differences[0] = self.km_calibration_.difference_at_t(
            times, inc_prob_at_conf[:, 0, :]
        )

        for event_id in self.event_ids_[1:]:
            inc_func_aj = self.aalen_johansen_sampler_.incidence_func_[event_id]
            inc_probs_aj = inc_func_aj(times)
            inc_probs_mean = inc_prob_at_conf[:, event_id, :].mean(axis=0)
            differences[event_id] = inc_probs_mean - inc_probs_aj

        return differences

    @staticmethod
    def _sort_by_time(times, preds_3d):
        """Return (sorted_times, preds_reordered) with ascending time order."""
        times = np.asarray(times)
        preds_3d = np.asarray(preds_3d)
        order = np.argsort(times)
        return times[order], preds_3d[:, :, order]


def aj_calibration(y_conf, times, inc_prob_at_conf, return_diff_at_t=False, alpha=2):
    r"""AJ-Calibration: marginal calibration score for competing risks models.

    Measures how closely the mean predicted cumulative incidence functions
    (CIFs) track the marginal Aalen-Johansen estimates. For each cause of
    event :math:`k` the score is:

    .. math::

        \text{AJ-Cal}_k = \frac{1}{t_{\max}} \int_0^{t_{\max}}
        \left(\bar{F}_k(t) - \hat{F}^{AJ}_k(t)\right)^\alpha \, dt

    where :math:`\bar{F}_k(t) = \frac{1}{n} \sum_{i=1}^n
    \hat{F}_k(t \mid \mathbf{x}_i)` is the mean predicted CIF and
    :math:`\hat{F}^{AJ}_k(t)` is the Aalen-Johansen CIF for event :math:`k`.

    The survival probability (event 0) is scored with
    :func:`~hazardous.metrics.km_calibration`.

    Parameters
    ----------
    y_conf : array-like of shape (n_samples, 2)
        Survival outcomes of the calibration cohort, with columns
        ``"event"`` (0 for censoring, positive integers for each cause of
        event) and ``"duration"`` (observed time).

    times : array-like of shape (n_times,)
        Time points at which the CIFs were predicted.

    inc_prob_at_conf : array-like of shape (n_samples, n_events+1, n_times)
        Predicted incidence probabilities at ``times`` for the calibration
        cohort. The second axis indexes event identifiers in sorted order.

    return_diff_at_t : bool, default=False
        If ``True``, also return pointwise differences
        :math:`\bar{F}_k(t) - \hat{F}^{AJ}_k(t)` for each event.

    alpha : int, default=2
        Exponent applied to the pointwise difference before integration.

    Returns
    -------
    aj_calibrations : dict of {int: float}
        AJ-Calibration score for each event identifier.

    differences_at_t : dict of {int: ndarray of shape (n_times,)}, optional
        Pointwise differences for each event identifier.
        Only returned when ``return_diff_at_t=True``.

    See Also
    --------
    AJCalibration : Class-based API.
    km_calibration : KM-Calibration for single-event survival.

    References
    ----------
    .. [Alberge2026] J. Alberge, T. Haugomat, G.Varoquaux,
        J. Abecassis,  "On the calibration of survival models with competing risks",
        arXiv:2602.00194, 2026.
        https://arxiv.org/pdf/2602.00194
    """
    cal = AJCalibration(alpha=alpha).fit(y_conf)
    scores = cal.score(times, inc_prob_at_conf)
    if return_diff_at_t:
        differences = cal.difference_at_t(times, inc_prob_at_conf)
        return scores, differences
    return scores
