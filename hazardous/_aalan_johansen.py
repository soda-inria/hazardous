import numpy as np
from lifelines import AalenJohansenFitter
from scipy.interpolate import interp1d
from sklearn.base import BaseEstimator, check_is_fitted

from hazardous.metrics._brier_score import (
    integrated_brier_score_incidence,
    integrated_brier_score_incidence_oracle,
)
from hazardous.utils import check_y_survival


class AalenJohansenEstimator(BaseEstimator):
    """Aalen Johasen competing risk estimator.

    Parameters
    ----------
    random_state : default=None
        Used to subsample X during fit when X has more samples
        than max_fit_samples.
    """

    def __init__(
        self,
        max_fit_samples=10_000,
        random_state=None,
        calculate_variance=False,
        seed=0,
    ):
        self.max_fit_samples = max_fit_samples
        self.random_state = random_state
        self.calculate_variance = calculate_variance
        self.seed = seed

    def fit(self, y):
        """
        Parameters
        ----------
        y : pandas.DataFrame of shape (n_samples, 2)
            The target, with columns 'event' and 'duration'.

        Returns
        -------
        self : fitted instance of FineGrayEstimator
        """
        self._check_input(y)
        event, duration = check_y_survival(y)

        self.times_ = np.unique(duration[event > 0])
        self.event_ids_ = np.array(sorted(list(set([0]) | set(event))))

        self.aj_fitter_events_ = []
        for event_id in self.event_ids_[1:]:
            aj_fitter_event = AalenJohansenFitter(
                calculate_variance=self.calculate_variance, seed=self.seed
            )
            aj_fitter_event.fit(duration, event, event_of_interest=event_id)

            self.aj_fitter_events_.append(aj_fitter_event)

        return self

    def predict_cumulative_incidence(self, X, times=None):
        """Predict the conditional cumulative incidence.

        Parameters
        ----------
        X : pd.DataFrame of shape (n_samples, n_features)
            Only used for its shape.

        times : ndarray of shape (n_times,), default=None
            The time steps to estimate the cumulative incidence at.
            * If set to None, the duration of the event of interest
              seen during fit 'times_' is used.
            * If not None, this performs a linear interpolation for each sample.

        Returns
        -------
        y_pred : ndarray of shape (n_samples, n_times)
            The conditional cumulative cumulative incidence at times.
        """
        check_is_fitted(self, "aj_fitter_events_")

        if times is None:
            times = self.times_

        all_y_pred = []
        for aj in self.aj_fitter_events_:
            # Interpolate each sample
            cif = aj.cumulative_density_
            times_event = cif.index
            y_pred = cif.values[:, 0]

            times_event = np.hstack([[0], times_event, [np.inf]])
            y_pred = np.hstack([[0], y_pred, [y_pred[-1]]])

            y_pred = interp1d(
                x=times_event,
                y=y_pred,
                kind="linear",
            )(times)

            all_y_pred.append(y_pred)

        surv_curve = (1 - np.sum(all_y_pred, axis=0))[None, :]

        y_pred = np.concatenate([surv_curve, all_y_pred], axis=0)
        y_pred = [y_pred for _ in range(X.shape[0])]

        return np.swapaxes(y_pred, 0, 1)

    def _check_input(self, y):
        if not hasattr(y, "__dataframe__"):
            raise TypeError(f"'y' must be a Pandas dataframe, got {type(y)}.")

    def score(self, X, y, shape_censoring=None, scale_censoring=None):
        """Return

        #TODO: implement time integrated NLL.
        """
        predicted_curves = self.predict_cumulative_incidence(X)
        ibs_events = []

        for idx, event_id in enumerate(self.event_ids_[1:]):
            predicted_curves_for_event = predicted_curves[idx]
            if scale_censoring is not None and shape_censoring is not None:
                ibs_event = integrated_brier_score_incidence_oracle(
                    y_train=self.y_train,
                    y_test=y,
                    y_pred=predicted_curves_for_event,
                    times=self.times_,
                    shape_censoring=shape_censoring,
                    scale_censoring=scale_censoring,
                    event_of_interest=event_id,
                )

            else:
                ibs_event = integrated_brier_score_incidence(
                    y_train=self.y_train,
                    y_test=y,
                    y_pred=predicted_curves_for_event,
                    times=self.times_,
                    event_of_interest=event_id,
                )

            ibs_events.append(ibs_event)
        return -np.mean(ibs_events)
