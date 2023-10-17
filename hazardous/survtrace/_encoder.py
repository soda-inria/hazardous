import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from hazardous.utils import check_y_survival


class SurvFeatureEncoder(TransformerMixin, BaseEstimator):
    """Apply standard scaling and ordinal encoding to the features \
    before tokenizing all categories together.

    Parameters
    ----------
    categ_cols : iterable of str, default=None
        The categorical column names of the input.
        If set to None, use dtypes to infer.

    num_cols : iterable of str, default=None
        The numerical column names of the input.
        If set to None, use dtypes to infer.

    Attributes
    ----------
    categ_cols_ : iterable of str
        The categorical column names of the input.

    num_cols_ : iterable of str
        The numerical column names of the input.

    col_transformer_ : :class:`~sklearn.compose.ColumnTransformer`
        Applies transformers to columns of an array or pandas DataFrame.

    vocab_size_ : ndarray of shape (n_categ_cols,)
        The cumulative sum of the cardinality of each categorical columns.
        Used to tokenize the categories across all columns using a shared
        vocabulary.

    Examples
    --------
    >>> X = pd.DataFrame([ \
            ["a", "c", 1],
            ["b", "d", 2],
            ["a", "e", 3],
        ])
    >>> SurvFeatureEncoder().fit_transform(X)
    array([
        [ 0., 2. , -1.22474487],
        [ 1., 3., 0.],
        [ 0., 4., 1.22474487],
    ])

    """

    def __init__(self, categ_cols=None, num_cols=None):
        self.categ_cols = categ_cols
        self.num_cols = num_cols

    def fit(self, X, y=None):
        self._check_num_categ_cols(X)

        self.col_transformer_ = make_column_transformer(
            (OrdinalEncoder(), self.categ_cols_),
            (StandardScaler(), self.num_cols_),
            remainder="drop",
            verbose_feature_names_out=False,
        )
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
        out_feature_names = self.col_transformer_.get_feature_names_out()
        categ_indices = np.isin(out_feature_names, self.categ_cols_).nonzero()[0]
        X_t[:, categ_indices] += self.vocab_size_

        return X_t

    def _check_num_categ_cols(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"X must be a Pandas dataframe. Got {X=!r}.")

        if self.num_cols is None:
            self.num_cols_ = X.select_dtypes("number").columns
        else:
            self.num_cols_ = np.atleast_1d(self.num_cols)

        if self.categ_cols is None:
            self.categ_cols_ = X.select_dtypes(["object", "string", "category"]).columns
        else:
            self.categ_cols_ = np.atleast_1d(self.categ_cols)


class SurvTargetEncoder(TransformerMixin, BaseEstimator):
    """Bin the durations using quantile horizons.

    Parameters
    ----------
    quantile_horizons : iterable of float, default=None
        The sequence of percentage used to create the time grid from the
        training durations. Between 0 and 1.
        If None, it is set to [.25, .5, .75].

    Attributes
    ----------
    quantile_horizons_ : iterable of float
        The sequence of percentage used to create the time grid from the
        training durations. Between 0 and 1.

    time_grid_ : list of float
        The list of horizons used to bin the duration
        The first value is 0 and the last if the max duration.

    time_grid_to_idx_ : mapping from float to int
        Mapping of each element of the time grid to its position.
    """

    def __init__(self, quantile_horizons=None):
        self.quantile_horizons = quantile_horizons

    def fit(self, y):
        """Create the time grid from the training durations.

        Parameters
        ----------
        y : pandas.DataFrame
            Input target, must contains an event and duration column.
        """
        event, duration = check_y_survival(y)

        if self.quantile_horizons is None:
            self.quantile_horizons_ = [0.25, 0.5, 0.75]
        else:
            self.quantile_horizons_ = self.quantile_horizons

        # Check strictly increasing
        if not np.all(self.quantile_horizons_[:-1] < self.quantile_horizons_[1:]):
            raise ValueError(
                f"'horizon' must be strictly increasing, got {self.horizons=!r}"
            )

        # Defining cuts
        mask_first_event = event == 1
        first_event_durations = duration[mask_first_event]
        time_grid = np.quantile(first_event_durations, self.quantile_horizons_).tolist()

        max_duration = duration.max()
        self.time_grid_ = np.array([0, *time_grid, max_duration])
        self.time_grid_to_idx_ = dict(zip(self.time_grid_, range(len(self.time_grid_))))

        return self

    def transform(self, y):
        """Apply the binning of durations.

        Parameters
        ----------
        y : pd.DataFrame
            Input target, must contains an event and duration column.

        Returns
        -------
        y_transformed : pd.DataFrame
            dataframe with 3 columns:
            - event (int): the input array of events updated with censoring
              from the edges of the discrete duration.
            - duration (int): discrete array of durations, represented
              by the index of the time grid.
            - frac_duration (float): the remaining duration between the
              left edge of the bin and the input, continuous duration.
        """
        event, duration = check_y_survival(y)

        # Apply right censoring when the duration is higher
        # than the last interval.
        max_cut = self.time_grid_.max()
        mask_censor = duration > max_cut
        duration[mask_censor] = max_cut
        event[mask_censor] = 0

        # Binning the durations
        bins = np.searchsorted(self.time_grid_, duration, side="left")
        binned_durations = self.time_grid_[bins]
        idx_durations = np.array(
            [self.time_grid_to_idx_[bin] for bin in binned_durations]
        )

        time_grid_diff = np.diff(self.time_grid_)
        frac_durations = (
            1.0 - (binned_durations - duration) / time_grid_diff[idx_durations - 1]
        )

        # Event or censoring at start time should be removed.
        # We left them at zero so that they don't contribute to the loss.
        mask_zero_duration = idx_durations == 0
        frac_durations[mask_zero_duration] = 0
        event[mask_zero_duration] = 0

        return pd.DataFrame(
            dict(
                event=event.astype("int64"),
                duration=idx_durations.astype("int64") - 1,
                frac_duration=frac_durations.astype("float32"),
            )
        )
