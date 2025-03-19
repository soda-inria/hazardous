import numpy as np
import pandas as pd
import torch
import torchtuples as tt
from pycox.models import DeepHit
from pycox.preprocessing.label_transforms import LabTransDiscreteTime
from sklearn.model_selection import train_test_split
from scipy.interpolate import interp1d

from hazardous.metrics._brier_score import (
    integrated_brier_score_incidence,
)

SEED = 0

np.random.seed(SEED)
_ = torch.manual_seed(SEED)


class LabTransform(LabTransDiscreteTime):
    def transform(self, durations, events):
        durations, is_event = super().transform(durations, events > 0)
        events[is_event == 0] = 0
        return durations, events.astype("int64")


class CauseSpecificNet(torch.nn.Module):
    """Network structure similar to the DeepHit paper, but without the residual
    connections (for simplicity).
    """

    def __init__(
        self,
        in_features,
        num_nodes_shared,
        num_nodes_indiv,
        num_risks,
        out_features,
        batch_norm=True,
        dropout=None,
    ):
        super().__init__()
        self.shared_net = tt.practical.MLPVanilla(
            in_features,
            num_nodes_shared[:-1],
            num_nodes_shared[-1],
            batch_norm,
            dropout,
        )
        self.risk_nets = torch.nn.ModuleList()
        for _ in range(num_risks):
            net = tt.practical.MLPVanilla(
                num_nodes_shared[-1],
                num_nodes_indiv,
                out_features,
                batch_norm,
                dropout,
            )
            self.risk_nets.append(net)

    def forward(self, input):
        out = self.shared_net(input)
        out = [net(out) for net in self.risk_nets]
        out = torch.stack(out, dim=1)
        return out


def get_target(df):
    return (
        df["duration"].astype("float32").values,
        df["event"].astype("int32").values,
    )


class DeepHitEstimator(tt.Model):
    def __init__(
        self,
        num_nodes_shared=[64, 64],
        num_nodes_indiv=[32],
        batch_size=256,
        epochs=512,
        callbacks=[tt.callbacks.EarlyStoppingCycle()],
        verbose=False,
        num_durations=10,
        batch_norm=True,
        dropout=None,
        alpha=0.2,
        sigma=0.1,
    ):
        self.num_durations = num_durations
        self.num_nodes_shared = num_nodes_shared
        self.num_nodes_indiv = num_nodes_indiv
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.alpha = alpha
        self.sigma = sigma
        self.batch_size = batch_size
        self.epochs = epochs
        self.callbacks = callbacks
        self.verbose = verbose

    def fit(self, X, y):
        if type(X) == pd.DataFrame:
            X_ = X.values.astype("float32")
        else:
            X_ = X.astype("float32")

        X_train, X_val, y_train, y_val = train_test_split(
            X_, y, test_size=0.2, random_state=SEED
        )

        self.labtrans = LabTransform(self.num_durations)

        y_train = self.labtrans.fit_transform(*get_target(y_train))
        y_val = self.labtrans.transform(*get_target(y_val))
        # TODO : REMOVE THIS
        self.y_train = y_train
        self.in_features = X_train.shape[1]
        self.num_risks = y_train[1].max()
        self.times_ = self.labtrans.cuts

        self.net = CauseSpecificNet(
            in_features=self.in_features,
            num_nodes_shared=self.num_nodes_shared,
            num_nodes_indiv=self.num_nodes_indiv,
            num_risks=self.num_risks,
            out_features=len(self.labtrans.cuts),
            batch_norm=self.batch_norm,
            dropout=self.dropout,
        )

        self.optimizer = tt.optim.AdamWR(
            lr=0.01, decoupled_weight_decay=0.01, cycle_eta_multiplier=0.8
        )

        self.model = DeepHit(
            net=self.net,
            optimizer=self.optimizer,
            alpha=self.alpha,
            sigma=self.sigma,
            duration_index=self.labtrans.cuts,
        )

        self.model.fit(
            X_train,
            y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            callbacks=self.callbacks,
            verbose=self.verbose,
            val_data=(X_val, y_val),
        )

    def _predict_survival_function(self, X):
        if type(X) == pd.DataFrame:
            X_ = X.values.astype("float32")
        else:
            X_ = X.astype("float32")
        return self.model.predict_surv_df(X_)

    def predict_survival_function(self, X, times=None):
        return self.predict_cumulative_incidence(X, times=times)[0]

    def predict_cumulative_incidence(self, X, times=None):
        if type(X) == pd.DataFrame:
            X_ = X.values.astype("float32")
        else:
            X_ = X.astype("float32")
        y_pred = self.model.predict_cif(X_)
        y_pred = np.swapaxes(y_pred, 0, 2)
        y_pred = np.swapaxes(y_pred, 1, 2)
        survival_pred = 1 - y_pred.sum(axis=1)
        survival_pred = survival_pred[:, None, :]
        y_pred = np.concatenate((survival_pred, y_pred), axis=1)

        all_event_y_pred = []

        for event in range(y_pred.shape[1]):
            event_pred = y_pred[:, event, :]
            if event == 0:
                y_pred_at_0 = np.ones((event_pred.shape[0], 1))
            else:
                y_pred_at_0 = np.zeros((event_pred.shape[0], 1))

            y_pred_t_max = event_pred[:, [-1]]
            event_pred = np.hstack([y_pred_at_0, event_pred, y_pred_t_max])

            if times is None:
                times = self.times_

            times_event = np.hstack([[0], self.times_, [np.inf]])

            all_y_pred = []
            for idx in range(event_pred.shape[0]):
                y_pred_ = interp1d(
                    x=times_event,
                    y=event_pred[idx, :],
                    kind="linear",
                )(times)
                all_y_pred.append(y_pred_)

            event_pred = np.vstack(all_y_pred)
            all_event_y_pred.append(event_pred)

        all_event_y_pred = np.asarray(all_event_y_pred)
        return all_event_y_pred.swapaxes(0, 1)

    def predict_proba(self, X):
        if type(X) == pd.DataFrame:
            X_ = X.values.astype("float32")
        else:
            X_ = X.astype("float32")
        return self.model.predict_pmf(X_)

    def score(self, X, y):
        if type(X) == pd.DataFrame:
            X_ = X.values.astype("float32")
        else:
            X_ = X.astype("float32")
        predicted_curves = self.model.predict_cif(X_)
        ibs_events = []
        for event in range(len(predicted_curves)):
            predicted_curves_for_event = predicted_curves[event]
            ibs_event = integrated_brier_score_incidence(
                y_train=self.y_train,
                y_test=y,
                y_pred=predicted_curves_for_event,
                times=self.labtrans.cuts,
                event_of_interest=event + 1,
            )

            ibs_events.append(ibs_event)
        return -np.mean(ibs_events)

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        del deep
        out = dict()
        out["num_durations"] = self.num_durations
        out["num_nodes_shared"] = self.num_nodes_shared
        out["num_nodes_indiv"] = self.num_nodes_indiv
        out["batch_norm"] = self.batch_norm
        out["dropout"] = self.dropout
        out["alpha"] = self.alpha
        out["sigma"] = self.sigma
        out["optimizer"] = self.optimizer
        out["batch_size"] = self.batch_size
        out["epochs"] = self.epochs
        out["callbacks"] = self.callbacks
        out["verbose"] = self.verbose
        return out

    def set_params(self, **parameters):
        """
        Set the parameters of this estimator.

        Returns
        -------
        self
        """
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
