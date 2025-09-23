# %%
import numpy as np
import torch

from hazardous.recalibration_posthoc.temperature_scaling import (
    InverseTemperatureScalingCalibrator,
)
from hazardous.utils import check_y_survival
from models_sota._aalen_johansen import AalenJohansenEstimator


class RecalibrationTS:
    """
    Recalibration using temperature scaling.
    This class is used to apply temperature scaling to the logits
    of a model's predictions.
    """

    def __init__(self, model, seed=0, X_aj=None, y_aj=None):
        self.model = model
        torch.manual_seed(seed)
        self.X_aj = X_aj
        self.y_aj = y_aj

    def fit(self, X_conf, y_conf, times=None, epsilon=1e-5):
        """
        Fit the temperature scaling model using the validation set.
        """
        X_conf = np.clip(X_conf, 1e-10, 1 - 1e-10)
        if times is None:
            times = self.model.time_grid_
        self.times = times
        event, duration = check_y_survival(y_conf)
        self.event_ids_ = np.array(sorted(list(set([0]) | set(event))))
        prediction_conf = self.model.predict_cumulative_incidence(
            X_conf, times=self.times
        )
        # Fit an Aalen-Johansen estimator on the validation set to get the targets
        if self.X_aj is not None:
            aj = AalenJohansenEstimator().fit(self.X_aj, self.y_aj)
        else:
            aj = AalenJohansenEstimator().fit(X_conf, y_conf)
        aj_preds_times = aj.predict_cumulative_incidence(X_conf, self.times)
        self.temperature = []
        for idx, tau in enumerate(self.times):
            predictions_tau = prediction_conf[:, :, idx]
            # compute the targets at time tau
            targets_tau = aj_preds_times[:, :, idx]

            model_with_temperature = InverseTemperatureScalingCalibrator()
            model_with_temperature.fit(predictions_tau, targets_tau)
            # import ipdb; ipdb.set_trace()
            self.temperature.append(
                (1.0 / model_with_temperature.inv_temperature_).item()
            )
        self.temperature = np.asarray(self.temperature)
        return self

    def _get_logits(self, X):
        X = X + 1e-10
        X /= np.sum(X, axis=-1, keepdims=True)
        return torch.as_tensor(np.log(X), dtype=torch.float32)

    def predict_cumulative_incidence(self, X, times=None, epsilon=1e-5):
        """
        Predict the recalibrated probabilities using the fitted
        temperature scaling model.
        """
        prediction = self.model.predict_cumulative_incidence(X, times=self.times)
        prediction = np.clip(prediction, 1e-10, 1 - 1e-10)
        prediction_logits = self._get_logits(prediction)
        prediction_logits_temps = prediction_logits / self.temperature[None, None, :]
        # apply softmax on the logits
        prediction_after_temp = (
            torch.softmax(prediction_logits_temps, dim=1).detach().numpy()
        )
        if times is None:
            return prediction_after_temp
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
                        prediction_after_temp[sample, event_id, :],
                        left=0,
                    )
            return recalibrated_probabities

    def compute_ft(self, X, y, epsilon=1e-5):
        prediction = self.model.predict_cumulative_incidence(X, times=self.times)
        prediction = np.clip(prediction, 1e-10, 1 - 1e-10)
        prediction_logits = self._get_logits(prediction)
        # / (prediction[:, 0, :][:, None, :] + epsilon)
        prediction_logits_temps = prediction_logits / self.temperature[None, None, :]
        # apply softmax on the logits
        prediction_after_temp = (
            torch.softmax(prediction_logits_temps, dim=1).detach().numpy()
        )

        recalibrated_probabities = np.zeros((X.shape[0], self.event_ids_.shape[0], 1))
        for sample in range(X.shape[0]):
            for event_id in self.event_ids_:
                # import ipdb; ipdb.set_trace()
                recalibrated_probabities[sample, event_id, :] = np.interp(
                    np.array([y["duration"].iloc[sample]]),
                    self.times,
                    prediction_after_temp[sample, event_id, :],
                    left=0,
                )
        return recalibrated_probabities
