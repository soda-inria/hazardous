import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


class FeaturePreprocessor(TransformerMixin, BaseEstimator):
    def __init__(self, categ_cols, num_cols):
        self.categ_cols = categ_cols
        self.num_cols = num_cols

    def fit(self, X, y=None):
        self.col_transformer_ = make_column_transformer(
            (OrdinalEncoder(), self.categ_cols),
            (StandardScaler(), self.num_cols),
            remainder="drop",
            verbose_feature_names_out=False,
        )
        self.col_transformer_.set_output(transform="pandas")
        self.col_transformer_.fit(X)

        # Encode categorical values from different columns separately
        # e.g. "blue" in column 1 is represented by a different token
        # than "blue" in column 2.
        transformers = self.col_transformer_.transformers_
        categories = transformers[0][1].categories_
        vocab_size = [0, *[len(categs) for categs in categories[:-1]]]
        self.vocab_size_ = np.cumsum(vocab_size)

        return self

    def transform(self, X, y=None):
        X_t = self.col_transformer_.transform(X)
        X_t[self.categ_cols] += self.vocab_size_

        return X_t


class TargetPreprocessor(TransformerMixin, BaseEstimator):
    def __init__(self, horizons):
        self.horizons = horizons

    def fit(self, y):
        # Check strictly increasing
        if not np.all(self.horizons[:-1] < self.horizons[1:]):
            raise ValueError(
                f"'horizon' must be strictly increasing, got {self.horizons=!r}"
            )

        # Defining cuts
        mask_first_event = y["event"] == 1
        first_event_durations = y.loc[mask_first_event]["duration"]
        times = np.quantile(first_event_durations, self.horizons).tolist()
        max_duration = y["duration"].max()
        self.cuts_ = np.array([0, *times, max_duration])
        self.cuts_to_idx_ = dict(zip(self.cuts_, range(len(self.cuts_))))

        return self

    def transform(self, y):
        durations = y["duration"]
        events = y["event"]

        # Apply right censoring when
        max_cut = self.cuts_.max()
        censor = durations > max_cut
        durations[censor] = max_cut
        events[censor] = 0

        # Binning the durations
        bins = np.searchsorted(self.cuts_, durations, side="left")
        binned_durations = self.cuts_[bins]
        idx_durations = np.array([self.cuts_to_idx_[bin] for bin in binned_durations])

        cut_diff = np.diff(self.cuts_)
        frac_durations = (
            1.0 - (binned_durations - durations) / cut_diff[idx_durations - 1]
        )

        # Event or censoring at start time should be removed.
        # We left them at zero so that they don't contribute to the loss.
        mask_zero_duration = idx_durations == 0
        frac_durations[mask_zero_duration] = 0
        events[mask_zero_duration] = 0

        return pd.DataFrame(
            dict(
                event=events.astype("int64"),
                duration=idx_durations.astype("int64") - 1,
                frac_duration=frac_durations.astype("float32"),
            )
        )
