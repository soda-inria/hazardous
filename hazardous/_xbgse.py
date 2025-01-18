import numpy as np
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split
from xgbse import XGBSEDebiasedBCE
from xgbse.converters import convert_to_structured

from hazardous.metrics._brier_score import integrated_brier_score_incidence


class XGBSE(XGBSEDebiasedBCE):
    def __init__(
        self,
        num_boost_round=10,
        early_stopping_rounds=None,
        lr_params=None,
        xgb_params=None,
        n_jobs=1,
        random_state=None,
    ):
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state = random_state
        if lr_params is None:
            lr_params = {}
        super().__init__(
            lr_params=lr_params,
            xgb_params=xgb_params,
            n_jobs=n_jobs,
        )

    def fit(self, X, y):
        y_rec = convert_to_structured(T=y["duration"], E=y["event"])

        if self.early_stopping_rounds is not None:
            X, X_val, y_rec, y_val = train_test_split(
                X, y_rec, test_size=0.2, random_state=self.random_state
            )
            validation_data = (X_val, y_val)
        else:
            validation_data = None
        return super().fit(
            X,
            y_rec,
            num_boost_round=self.num_boost_round,
            validation_data=validation_data,
            early_stopping_rounds=self.early_stopping_rounds,
        )

    def predict_cumulative_incidence(self, X, times=None):
        y_surv_proba = self.predict(X).to_numpy()

        if times is not None:
            interpolated = []
            for idx in range(y_surv_proba.shape[0]):
                interpolated.append(
                    interp1d(
                        x=self.time_grid_,
                        y=y_surv_proba[idx, :],
                        kind="linear",
                        bounds_error=False,
                        fill_value="extrapolate",
                        assume_sorted=True,
                    )(times)[None, :]
                )
            y_surv_proba = np.concatenate(interpolated, axis=0)

        y_proba = np.concatenate(
            [y_surv_proba[:, None, :], (1 - y_surv_proba)[:, None, :]], axis=1
        )
        return y_proba

    def score(self, X, y):
        y_proba = self.predict_cumulative_incidence(X)

        ibs = integrated_brier_score_incidence(
            y_train=y,
            y_test=y,
            y_pred=y_proba[:, 1, :],
            times=self.time_grid_,
            event_of_interest=1,
        )
        return -ibs

    @property
    def time_grid_(self):
        return self.time_bins
