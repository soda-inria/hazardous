import numpy as np
from scipy.interpolate import interp1d

from .._km_sampler import _KaplanMeierSampler


class KMCalibration:
    r"""Marginal calibration of a survival model using the Kaplan-Meier estimator.

    This class measures how closely the mean predicted survival function
    matches the Kaplan-Meier marginal estimate over a given time grid. The
    calibration score is defined as:

    .. math::

        \text{KM-Cal} = \frac{1}{t_{\max}} \int_0^{t_{\max}}
        \left(\bar{S}(t) - \hat{S}_{KM}(t)\right)^\alpha \, dt

    where :math:`\bar{S}(t)` is the mean predicted survival probability across
    the calibration cohort and :math:`\hat{S}_{KM}(t)` is the Kaplan-Meier
    estimate fitted on that same cohort.

    A score of zero indicates perfect marginal calibration. Positive values
    indicate systematic over- or under-prediction of survival.

    The KM-Calibration score is a special case of the Aalen-Johansen calibration
    score for a single event type. The implementation of this metric is also
    available in SurvivalEval[Qi2024], a python library for survival analysis evaluation
    metrics.
    Parameters
    ----------
    alpha : int, default=2
        Exponent applied to the pointwise difference before integration.
        When ``alpha=2``, the score is an L2 (squared) calibration score.
        When ``alpha=1``, it is an L1 (signed absolute) calibration score.

    Attributes
    ----------
    kaplan_meier_sampler_ : _KaplanMeierSampler
        The fitted Kaplan-Meier sampler used to estimate the marginal survival
        function on the calibration cohort.

    See Also
    --------
    km_calibration : Functional API for this class.
    AJCalibration : Extends calibration to competing risks via Aalen-Johansen.

    References
    ----------
    .. [Alberge2026] J. Alberge, T. Haugomat, G.Varoquaux,
        J. Abecassis,  "On the calibration of survival models with competing risks",
        arXiv:2602.00194, 2026.
        https://arxiv.org/pdf/2602.00194

    .. [Qi2024] S. Qi, W. Sun, R. Greiner.
    "{SurvivalEVAL}: A Comprehensive Open-Source Python Package
    for Evaluating Individual Survival Distributions." 10.1609/aaaiss.v2i1.27713 (2024).
    """

    def __init__(self, alpha=2):
        self.alpha = alpha

    def fit(self, y_conf):
        """Fit the Kaplan-Meier estimator on the calibration cohort.

        Parameters
        ----------
        y_conf : array-like of shape (n_samples, 2)
            Survival outcomes of the calibration cohort, with columns
            ``"event"`` (0 for censoring, 1 for the event) and
            ``"duration"`` (observed time).

        Returns
        -------
        self : object
            Fitted estimator.
        """
        self.kaplan_meier_sampler_ = _KaplanMeierSampler().fit(y_conf)
        return self

    def score(self, times, surv_prob_at_conf):
        """Compute the KM-Calibration score.

        Parameters
        ----------
        times : array-like of shape (n_times,)
            Time points at which the survival probability was predicted.
            Need not be sorted; columns of ``surv_prob_at_conf`` must
            correspond to the same ordering as ``times``.

        surv_prob_at_conf : array-like of shape (n_samples, n_times)
            Predicted survival probabilities at ``times`` for the calibration
            cohort.

        Returns
        -------
        score : float
            KM-Calibration score. A value of 0 indicates perfect marginal
            calibration.
        """
        times, surv_prob_at_conf = self._sort_by_time(times, surv_prob_at_conf)
        t_max = times[-1]

        surv_probs_km = self.kaplan_meier_sampler_.survival_func_(times)
        surv_probs_mean = surv_prob_at_conf.mean(axis=0)

        diff_at_t = surv_probs_mean - surv_probs_km
        return np.trapezoid(np.abs(diff_at_t) ** self.alpha, times) / t_max

    def difference_at_t(self, times, surv_prob_at_conf):
        """Compute the pointwise difference between mean predictions and KM.

        Parameters
        ----------
        times : array-like of shape (n_times,)
            Time points at which the survival probability was predicted.

        surv_prob_at_conf : array-like of shape (n_samples, n_times)
            Predicted survival probabilities at ``times`` for the calibration
            cohort.

        Returns
        -------
        diff_at_t : ndarray of shape (n_times,)
            Pointwise difference :math:`\\bar{S}(t) - \\hat{S}_{KM}(t)`,
            returned in ascending time order.
        """
        times, surv_prob_at_conf = self._sort_by_time(times, surv_prob_at_conf)
        surv_probs_km = self.kaplan_meier_sampler_.survival_func_(times)
        surv_probs_mean = surv_prob_at_conf.mean(axis=0)
        return surv_probs_mean - surv_probs_km

    @staticmethod
    def _sort_by_time(times, preds_2d):
        """Return (sorted_times, preds_reordered) with ascending time order."""
        times = np.asarray(times)
        preds_2d = np.asarray(preds_2d)
        order = np.argsort(times)
        return times[order], preds_2d[:, order]


def km_calibration(y_conf, times, surv_prob_at_conf, return_diff_at_t=False, alpha=2):
    r"""KM-Calibration: marginal calibration score for survival models.

    Measures how closely the mean predicted survival probability tracks the
    Kaplan-Meier marginal estimate. The score is:

    .. math::

        \text{KM-Cal} = \frac{1}{t_{\max}} \int_0^{t_{\max}}
        \left(\bar{S}(t) - \hat{S}_{KM}(t)\right)^\alpha \, dt

    where :math:`\bar{S}(t) = \frac{1}{n} \sum_{i=1}^n
    \hat{S}(t \mid \mathbf{x}_i)` is the mean predicted survival probability
    and :math:`\hat{S}_{KM}(t)` is the Kaplan-Meier estimate fitted on the
    calibration cohort.

    Parameters
    ----------
    y_conf : array-like of shape (n_samples, 2)
        Survival outcomes of the calibration cohort, with columns
        ``"event"`` (0 for censoring, 1 for the event) and
        ``"duration"`` (observed time).

    times : array-like of shape (n_times,)
        Time points at which the survival probability was predicted.

    surv_prob_at_conf : array-like of shape (n_samples, n_times)
        Predicted survival probabilities at ``times`` for the calibration
        cohort.

    return_diff_at_t : bool, default=False
        If ``True``, also return the pointwise difference
        :math:`\bar{S}(t) - \hat{S}_{KM}(t)` at each time in ``times``.

    alpha : int, default=2
        Exponent applied to the pointwise difference before integration.
        When ``alpha=2``, the score is squared (L2 calibration).

    Returns
    -------
    km_cal : float
        KM-Calibration score. A value of 0 indicates perfect marginal
        calibration.

    diff_at_t : ndarray of shape (n_times,), optional
        Pointwise difference :math:`\bar{S}(t) - \hat{S}_{KM}(t)`.
        Only returned when ``return_diff_at_t=True``.

    See Also
    --------
    KMCalibration : Class-based API.
    aj_calibration : Extends to competing risks via Aalen-Johansen.

    References
    ----------
    .. [Alberge2026] J. Alberge, T. Haugomat, G.Varoquaux,
        J. Abecassis,  "On the calibration of survival models with competing risks",
        arXiv:2602.00194, 2026.
        https://arxiv.org/pdf/2602.00194
    """
    cal = KMCalibration(alpha=alpha).fit(y_conf)
    km_cal = cal.score(times, surv_prob_at_conf)
    if return_diff_at_t:
        diff_at_t = cal.difference_at_t(times, surv_prob_at_conf)
        return km_cal, diff_at_t
    return km_cal


def recalibrate_survival_function(
    X_conf,
    y_conf,
    times,
    estimator=None,
    X=None,
    surv_probs=None,
    surv_probs_conf=None,
    return_function=False,
):
    r"""Post-hoc recalibration of a survival function using KM-Calibration.

    Applies a marginal shift correction to survival probability estimates by
    subtracting the pointwise calibration error measured on a held-out
    calibration set:

    .. math::

        \tilde{S}(t \mid \mathbf{x}_i) =
        \hat{S}(t \mid \mathbf{x}_i) - \Delta(t)

    where :math:`\Delta(t) = \bar{S}(t) - \hat{S}_{KM}(t)` is the
    calibration error estimated on the calibration cohort.

    Either ``estimator`` or both ``surv_probs`` and ``surv_probs_conf`` must
    be provided.

    Parameters
    ----------
    X_conf : array-like of shape (n_conf, n_features)
        Feature matrix for the calibration cohort.

    y_conf : array-like of shape (n_conf, 2)
        Survival outcomes of the calibration cohort, with columns
        ``"event"`` and ``"duration"``.

    times : array-like of shape (n_times,)
        Time grid at which the survival function is evaluated.

    estimator : estimator object, default=None
        A fitted survival estimator with a ``predict_survival_function``
        method. Used to generate predictions when ``surv_probs`` is
        ``None``.

    X : array-like of shape (n_samples, n_features), default=None
        Feature matrix for the test set. Required when ``estimator`` is
        provided.

    surv_probs : array-like of shape (n_samples, n_times), default=None
        Pre-computed survival probability predictions for the test set at
        ``times``.

    surv_probs_conf : array-like of shape (n_conf, n_times), default=None
        Pre-computed survival probability predictions for the calibration
        cohort at ``times``.

    return_function : bool, default=False
        If ``True``, return a step-function interpolator instead of an
        array.

    Returns
    -------
    surv_probs_calibrated : ndarray of shape (n_samples, n_times) or callable
        Recalibrated survival probabilities, or an interpolation function
        when ``return_function=True``.

    See Also
    --------
    km_calibration : Compute the KM-Calibration score.
    recalibrate_survival_function_predictions : Recalibrate from predictions.

    References
    ----------
    .. [Alberge2026] J. Alberge, T. Haugomat, G.Varoquaux,
        J. Abecassis,  "On the calibration of survival models with competing risks",
        arXiv:2602.00194, 2026.
        https://arxiv.org/pdf/2602.00194
    """
    if estimator is None and (surv_probs is None or surv_probs_conf is None):
        raise ValueError(
            "Either estimator or (surv_probs and surv_probs_conf) must be provided."
        )

    if surv_probs is None:
        if not hasattr(estimator, "predict_survival_function"):
            raise ValueError("estimator must have a predict_survival_function method.")
        surv_probs = estimator.predict_survival_function(X, times)
        surv_probs_conf = estimator.predict_survival_function(X_conf, times)

    _, diff_at_t = km_calibration(y_conf, times, surv_probs_conf, return_diff_at_t=True)
    surv_probs_calibrated = np.asarray(surv_probs) - diff_at_t

    if return_function:
        return interp1d(
            x=times,
            y=surv_probs_calibrated,
            kind="previous",
            bounds_error=False,
            fill_value="extrapolate",
        )
    return surv_probs_calibrated


def recalibrate_survival_function_predictions(
    surv_probs,
    surv_probs_conf,
    y_conf,
    times,
    return_function=False,
):
    r"""Post-hoc recalibration from pre-computed survival predictions.

    Applies a marginal shift correction to survival probability estimates by
    subtracting the pointwise calibration error measured on a held-out
    calibration set:

    .. math::

        \tilde{S}(t \mid \mathbf{x}_i) =
        \hat{S}(t \mid \mathbf{x}_i) - \Delta(t)

    where :math:`\Delta(t) = \bar{S}(t) - \hat{S}_{KM}(t)` is the
    calibration error estimated on the calibration cohort.

    Parameters
    ----------
    surv_probs : array-like of shape (n_samples, n_times)
        Pre-computed survival probability predictions for the test set at
        ``times``.

    surv_probs_conf : array-like of shape (n_conf, n_times)
        Pre-computed survival probability predictions for the calibration
        cohort at ``times``.

    y_conf : array-like of shape (n_conf, 2)
        Survival outcomes of the calibration cohort, with columns
        ``"event"`` and ``"duration"``.

    times : array-like of shape (n_times,)
        Time grid at which the survival function is evaluated.

    return_function : bool, default=False
        If ``True``, return a step-function interpolator instead of an
        array.

    Returns
    -------
    surv_probs_calibrated : ndarray of shape (n_samples, n_times) or callable
        Recalibrated survival probabilities, or an interpolation function
        when ``return_function=True``.

    See Also
    --------
    km_calibration : Compute the KM-Calibration score.
    recalibrate_survival_function : Recalibrate from an estimator object.

    References
    ----------
    .. [Alberge2026] J. Alberge, T. Haugomat, G.Varoquaux,
        J. Abecassis,  "On the calibration of survival models with competing risks",
        arXiv:2602.00194, 2026.
        https://arxiv.org/pdf/2602.00194
    """
    _, diff_at_t = km_calibration(y_conf, times, surv_probs_conf, return_diff_at_t=True)
    surv_probs_calibrated = np.asarray(surv_probs) - diff_at_t

    if return_function:
        return interp1d(
            x=times,
            y=surv_probs_calibrated,
            kind="previous",
            bounds_error=False,
            fill_value="extrapolate",
        )
    return surv_probs_calibrated
