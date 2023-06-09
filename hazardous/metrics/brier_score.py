import numpy as np
from sklearn.utils.validation import check_random_state

from .._ipcw import IpcwEstimator


class BrierScoreComputer:
    def __init__(
        self,
        y_train,
        event_of_interest="any",
        random_state=None,
    ):
        if event_of_interest != "any" and event_of_interest < 1:
            raise ValueError(
                "event_of_interest must be a strictly positive integer or 'any', "
                f"got: event_of_interest={self.event_of_interest:!r}"
            )
        self.y_train = y_train
        self.y_train_any_event = self._any_event(y_train)
        self.event_of_interest = event_of_interest
        self.rng = check_random_state(random_state)

        # Estimate the censoring distribution from the training set
        # using Kaplan-Meier.
        self.ipcw_est = IpcwEstimator().fit(self.y_train_any_event)

        # Precompute the censoring probabilities at the time of the events on the
        # training set:
        self.ipcw_y_train = self.ipcw_est.predict(y_train["duration"])

    def _any_event(self, y):
        y_any_event = np.empty(
            y.shape[0],
            dtype=[("event", bool), ("duration", float)],
        )
        y_any_event["event"] = y["event"] > 0
        y_any_event["duration"] = y["duration"]
        return y_any_event

    def _ibs_components(self, y, times, ipcw_y):
        if self.event_of_interest == "any":
            # y should already be provided as binary indicator
            k = 1
        else:
            k = self.event_of_interest

        # Specify the binary classification target for each record in y and
        # a reference time horizon:
        #
        # - 1 when event of interest was observed before the reference time
        #   horizon,
        #
        # - 0 otherwise: any other event happening at any time, censored record
        #   or event of interest happening after the reference time horizon.
        #
        #   Note: censored events only contribute (as negative target) when
        #   their duration is larger than the reference target horizon.
        #   Otherwise, they are discarded by setting their weight to 0 in the
        #   following.

        y_binary = np.zeros(y.shape[0], dtype=np.int32)
        y_binary[(y["event"] == k) & (y["duration"] <= times)] = 1

        # Compute the weights for each term contributing to the Brier score
        # at the specified time horizons.
        #
        # - error of a prediction for a time horizon before the occurence of an
        #   event (either censored or uncensored) is weighted by the inverse
        #   probability of censoring at that time horizon.
        #
        # - error of a prediction for a time horizon after the any observed event
        #   is weighted by inverse censoring probability at the actual time
        #   of the observed event.
        #
        # - "error" of a prediction for a time horizon after a censored event has
        #   0 weight and do not contribute to the Brier score computation.

        # Estimate the probability of censoring at current time point t.
        ipcw_t = self.ipcw_est.predict(times)
        before = times < y["duration"]
        weights = np.where(before, ipcw_t, 0)

        after_any_observed_event = (y["event"] > 0) & (times >= y["duration"])
        weights = np.where(after_any_observed_event, ipcw_y, weights)

        return y_binary, weights

    def brier_score(self, y_true, y_pred, times):
        if self.event_of_interest == "any":
            if y_true is self.y_train:
                y_true = self.y_train_any_event
            else:
                y_true = self._any_event(y_true)

        n_samples = y_true.shape[0]
        n_time_steps = times.shape[0]
        brier_scores = np.empty(
            shape=(n_samples, n_time_steps),
            dtype=np.float64,
        )
        ipcw_y = self.ipcw_est.predict(y_true["duration"])
        for t_idx, t in enumerate(times):
            y_true_binary, weights = self._ibs_components(
                y=y_true,
                times=np.full(shape=n_samples, fill_value=t),
                ipcw_y=ipcw_y,
            )
            squared_error = (y_true_binary - y_pred[:, t_idx]) ** 2
            brier_scores[:, t_idx] = weights * squared_error

        return brier_scores.mean(axis=0)


def cif_brier_score(
    y_train,
    y_test,
    cif_pred,
    times,
    event_of_interest="any",
):
    # XXX: make times an optional kwarg to be compatible with
    # sksurv.metrics.brier_score?
    ibsts = BrierScoreComputer(
        y_train,
        event_of_interest=event_of_interest,
    )
    return times, ibsts.brier_score(y_test, cif_pred, times)


def cif_integrated_brier_score(
    y_train,
    y_test,
    cif_pred,
    times,
    event_of_interest="any",
):
    times, brier_scores = cif_brier_score(
        y_train,
        y_test,
        cif_pred,
        times,
        event_of_interest=event_of_interest,
    )
    return np.trapz(brier_scores, times) / (times[-1] - times[0])
