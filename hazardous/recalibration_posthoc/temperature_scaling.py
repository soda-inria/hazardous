import numpy as np
import torch
import torch.autograd as autograd
from sklearn.base import BaseEstimator
from torch import nn


# predictions_tau[torch.isnan(model_with_temperature
# ._get_logits(predictions_tau)).sum(dim=1)>0,:]
class InverseTemperatureScalingCalibrator(BaseEstimator):
    # following https://github.com/gpleiss/temperature_scaling/
    # blob/master/temperature_scaling.py
    def _get_logits(self, X):
        X = X + 1e-10
        X /= np.sum(X, axis=-1, keepdims=True)
        return torch.as_tensor(np.log(X), dtype=torch.float32)

    def fit(self, X, y):
        # X should be the probabilities as output by predict_proba()
        # clip X to avoid log(0)
        X = np.clip(X, 1e-10, 1 - 1e-10)
        logits = self._get_logits(X)
        labels = torch.as_tensor(y, dtype=torch.float32)
        self.inv_temperature_ = nn.Parameter(torch.ones(1))
        criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam([self.inv_temperature_], lr=0.01)

        if torch.isnan(logits).any():
            print("Error: Input logits contain NaN values.")
            return
        if torch.isinf(logits).any():
            print("Error: Input labels contain  Inf values.")
            return

        def eval():
            optimizer.zero_grad()
            y_pred = logits * torch.clamp(self.inv_temperature_, min=1e-3)
            loss = criterion(y_pred, labels)

            if torch.isnan(loss).any() or torch.isinf(loss).any():
                print(
                    f"Loss became NaN or Inf at iteration {i}. Stopping optimization."
                )
                print(f"Current inverse temperature: {self.inv_temperature_.item()}")
                return loss

            loss.backward()
            return loss

        with autograd.set_detect_anomaly(True):
            for i in range(50):
                optimizer.step(eval)

        print(f"Optimal temperature: {(1./self.inv_temperature_).item():g}")
        return self

    def predict_proba(self, X):
        # X should be the probabilities as output by predict_proba()
        logits = self._get_logits(X)
        with torch.no_grad():
            y_pred = logits * self.inv_temperature_[:, None]
            return torch.softmax(y_pred, dim=-1).detach().numpy()
