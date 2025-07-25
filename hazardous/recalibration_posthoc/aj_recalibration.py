import numpy as np

from hazardous.calibration._aj_calibration import aj_calibration
from hazardous.utils import check_y_survival


class RecalibrationAJ:
    """
    Recalibration using Aalen-Johansen calibration.
    This class is used to apply Aalen-Johansen calibration to the
    cumulative incidence functions of a model's predictions.
    """

    def __init__(self, model, seed=0):
        self.model = model
        self.seed = seed

    def fit(self, X_conf, y_conf, times=None):
        """
        Fit the Aalen-Johansen calibration model using the validation set.
        """
        if times is None:
            times = self.model.time_grid_
        self.times = times
        event, duration = check_y_survival(y_conf)
        self.event_ids_ = np.array(sorted(list(set([0]) | set(event))))

        if not hasattr(self.model, "predict_cumulative_incidence"):
            raise ValueError(
                "Estimator must have a predict_cumulative_incidence method"
            )

        inc_prob_conf = self.model.predict_cumulative_incidence(X_conf, times)
        # Calculate the survival probabilities to compute the calibration
        # Calculate the calibration
        self.differences_at_t = aj_calibration(
            y_conf, times, inc_prob_conf, return_diff_at_t=True
        )[1]
        return self

    def predict_cumulative_incidence(self, X, times=None):
        """
        Predict the recalibrated cumulative incidence functions using the fitted
        Aalen-Johansen calibration model.
        """
        incidence_probabilities = self.model.predict_cumulative_incidence(X, self.times)
        print("hello")
        recalibrated_inc_probabilities = []
        for event_id in self.event_ids_:
            diff_at_t = self.differences_at_t[event_id]
            # import ipdb; ipdb.set_trace()
            inc_probs_calibrated = (
                incidence_probabilities[:, event_id, :] - diff_at_t[None, :]
            )
            recalibrated_inc_probabilities.append(inc_probs_calibrated)

        recalibrated_inc_probabilities = np.array(
            recalibrated_inc_probabilities
        ).swapaxes(0, 1)
        recalibrated_inc_probabilities = np.clip(recalibrated_inc_probabilities, 0, 1)
        if times is None:
            return recalibrated_inc_probabilities
        else:
            # Interpolate the recalibrated probabilities to the requested times
            times = np.sort(times)
            recalibrated_probabities = np.zeros(
                (X.shape[0], self.event_ids_.shape[0], times.shape[0])
            )
            for sample in range(X.shape[0]):
                for event_id in self.event_ids_:
                    # import ipdb; ipdb.set_trace()
                    recalibrated_probabities[sample, event_id, :] = np.interp(
                        times,
                        self.times,
                        recalibrated_inc_probabilities[sample, event_id, :],
                    )

            return recalibrated_probabities

    def compute_ft(self, X, y, epsilon=1e-5):
        incidence_probabilities = self.model.predict_cumulative_incidence(X, self.times)
        print("hello")
        recalibrated_inc_probabilities = []
        for event_id in self.event_ids_:
            diff_at_t = self.differences_at_t[event_id]
            # import ipdb; ipdb.set_trace()
            inc_probs_calibrated = (
                incidence_probabilities[:, event_id, :] - diff_at_t[None, :]
            )
            recalibrated_inc_probabilities.append(inc_probs_calibrated)

        recalibrated_inc_probabilities = np.array(
            recalibrated_inc_probabilities
        ).swapaxes(0, 1)
        recalibrated_inc_probabilities = np.clip(recalibrated_inc_probabilities, 0, 1)

        # Interpolate the recalibrated probabilities to the requested times
        recalibrated_probabities = np.zeros((X.shape[0], self.event_ids_.shape[0], 1))
        for sample in range(X.shape[0]):
            for event_id in self.event_ids_:
                # import ipdb; ipdb.set_trace()
                recalibrated_probabities[sample, event_id, :] = np.interp(
                    np.array([y["duration"].iloc[sample]]),
                    self.times,
                    recalibrated_inc_probabilities[sample, event_id, :],
                )

        return recalibrated_probabities
