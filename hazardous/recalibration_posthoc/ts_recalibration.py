# %%
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from hazardous.recalibration_posthoc.temperature_scaling import ModelWithTemperature
from hazardous.utils import check_y_survival
from models_sota._aalen_johansen import AalenJohansenEstimator


class RecalibrationTS:
    """
    Recalibration using temperature scaling.
    This class is used to apply temperature scaling to the logits
    of a model's predictions.
    """

    def __init__(self, model, seed=0):
        self.model = model
        torch.manual_seed(seed)

    def fit(self, X_conf, y_conf, times=None, epsilon=1e-5):
        """
        Fit the temperature scaling model using the validation set.
        """
        if times is None:
            times = self.model.time_grid_
        self.times = times
        event, duration = check_y_survival(y_conf)
        self.event_ids_ = np.array(sorted(list(set([0]) | set(event))))
        prediction_conf = self.model.predict_cumulative_incidence(
            X_conf, times=self.times
        )
        prediction_conf_logits = np.log(
            (prediction_conf + epsilon)
        )  #  / (prediction_conf[:, 0, :][:, None, :]+ epsilon)
        # prediction_conf_logits = np.swapaxes(
        # np.swapaxes(prediction_conf_logits, 0, 1), 1, 2)

        aj = AalenJohansenEstimator().fit(X_conf, y_conf)
        aj_preds_times = aj.predict_cumulative_incidence(X_conf, self.times)
        self.temperature = []
        for idx, tau in enumerate(self.times):
            logits_tau = prediction_conf_logits[:, :, idx]
            # compute the targets at time tau
            targets_tau = aj_preds_times[:, :, idx]

            # Create a DataLoader for the validation set
            valid_dataset = TensorDataset(
                torch.tensor(X_conf.values, dtype=torch.float32),
                torch.tensor(targets_tau, dtype=torch.float32),
            )
            valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False)
            model_with_temperature = ModelWithTemperature()
            model_with_temperature.set_temperature(valid_loader, logits_tau)
            # import ipdb; ipdb.set_trace()
            self.temperature.append(model_with_temperature.temperature.item())
        self.temperature = np.asarray(self.temperature)
        return self

    def predict_cumulative_incidence(self, X, times=None, epsilon=1e-5):
        """
        Predict the recalibrated probabilities using the fitted
        temperature scaling model.
        """
        prediction = self.model.predict_cumulative_incidence(X, times=self.times)
        prediction_logits = np.log(
            (prediction + epsilon)
        )  # / (prediction[:, 0, :][:, None, :] + epsilon)
        prediction_logits_temps = prediction_logits / self.temperature[None, None, :]
        # apply softmax on the logits
        prediction_after_temp = np.exp(prediction_logits_temps) / np.sum(
            np.exp(prediction_logits_temps), axis=1, keepdims=True
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
        prediction_logits = np.log(
            (prediction + epsilon)
        )  # / (prediction[:, 0, :][:, None, :] + epsilon)
        prediction_logits_temps = prediction_logits / self.temperature[None, None, :]
        # apply softmax on the logits
        prediction_after_temp = np.exp(prediction_logits_temps) / np.sum(
            np.exp(prediction_logits_temps), axis=1, keepdims=True
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
