"""Helper to reproduce the SEER dataset transformations as performed in SurvTRACE.

See: https://github.com/RyanWangZf/SurvTRACE/blob/main/data/process_seer.py
"""

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, check_is_fitted
from sklearn.compose import make_column_transformer
from sklearn.datasets._base import Bunch
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

CATEGORICAL_COLUMN_NAMES = [
    "Sex",
    "Year of diagnosis",  # XXX shall this be typed as a numerical feature instead?
    "Race recode (W, B, AI, API)",
    "Histologic Type ICD-O-3",
    "Laterality",
    "Sequence number",
    "ER Status Recode Breast Cancer (1990+)",
    "PR Status Recode Breast Cancer (1990+)",
    "Summary stage 2000 (1998-2017)",
    "RX Summ--Surg Prim Site (1998+)",
    "Reason no cancer-directed surgery",
    "First malignant primary indicator",
    "Diagnostic Confirmation",
    "Median household income inflation adj to 2019",
]

NUMERIC_COLUMN_NAMES = [
    "Regional nodes examined (1988+)",
    "CS tumor size (2004-2015)",
    "Total number of benign/borderline tumors for patient",
    "Total number of in situ/malignant tumors for patient",
]

# "Unused x" names are placeholders, these columns are not used by SurvTRACE.
COLUMN_NAMES = [
    "Patient ID",
    "Sex",
    "Year of diagnosis",
    "Site recode ICD-0-3/WHO 2008",
    "COD to site recode",
    "Race recode (W, B, AI, API)",
    "Histologic Type ICD-O-3",
    "ICD-O-3 Hist/behav, malignant",
    "Laterality",
    "Sequence number",
    "Vital status recode (study cutoff used)",
    "ER Status Recode Breast Cancer (1990+)",
    "PR Status Recode Breast Cancer (1990+)",
    "Regional nodes examined (1988+)",
    "Unused 1",  # TODO: fixme: find informative column name
    "Summary stage 2000 (1998-2017)",
    "Unused 2",  # TODO: fixme: find informative column name
    "Reason no cancer-directed surgery",
    "CS tumor size (2004-2015)",
    "First malignant primary indicator",
    "Diagnostic Confirmation",
    "Total number of benign/borderline tumors for patient",
    "Total number of in situ/malignant tumors for patient",
    "Median household income inflation adj to 2019",
    "RX Summ--Surg Prim Site (1998+)",
    "SEER other cause of death classification",
    "SEER cause-specific death classification",
    "Survival months",
    "Unused 3",  # TODO: fixme: find informative column name
]


def load_seer(
    input_path,
    event_column_name="COD to site recode",
    duration_column_name="Survival months",
    events_of_interest=(
        "Breast",
        "Diseases of Heart",
    ),
    censoring_labels=("Alive",),
    other_event_name="Other",
    survtrace_preprocessing=False,
    return_X_y=False,
):
    """Load the seer dataset and optionally apply the same preprocessing \
        as done in SurvTRACE.

    The file is expected to be a txt file.

    Parameters
    ----------
    input_path : str or file_path
        The path of the txt file.

    events_of_interest : tuple of str or "all", \
            default=("Breast", "Diseases of Heart")
        If "all": all event types are preserved. Other specificy the labels of
        the event of interest to extract. All other events are collapsed into
        an "Other" event with a dedicated integer event code.

    censoring_labels : typle of str, default=("Alive",)
        The label(s) used in the COD (cause of death) column that should be
        interpreted as censoring marker(s) in the original dataset.

    other_event_name : str, default="Other"
        Whe other_events is "collapse", this parameter controls the name of the
        collapsed competing event.

    survtrace_preprocessing : bool, default=False
        If set to True, apply the preprocessing steps used in SurvTRACE to
        ensure reproducibility of the results in its paper. Note that to fully
        replicate the preprocessing used by SurvTRACE, one would also need to
        recode the "Other" competing event as 0 to treat it as censoring.

    Returns
    -------
    bunch_dataset : a Bunch object with the following attributes:

        data : pandas.DataFrame of shape (n_samples, n_features)
            The dataframe of features.

        target : pandas.DataFrame of shape (n_samples, 2)
            The two columns are named "event" and "duration".

            * The "event" columns holds integer identifiers of event of
            interest or 0 for censoring. The meaning of event integer codes
            is defined by the position in the event_labels list.
            * The "duration" columns holds a numerical value for the event
            free duration expressed in months.

            TODO: document what t0 mean.

        event_labels : list of str
            The labels of the events.

        original_data : pandas.DataFrame of shape (n_samples, 29)
            The original data.
    """
    msg = (
        f"The SEER dataset file doesn't exist at {input_path}."
        "See the installation guide at "
        "https://soda-inria.github.io/hazardous/downloading_seer.html"
    )
    if not Path(input_path).exists():
        raise FileNotFoundError(msg)

    # XXX: specify dtypes for all columns at load time insted of retyping a
    # posteriori?
    # In particular this should help remove a pandas warning for column 22.
    original_data = X = pd.read_csv(
        input_path, sep="\t", header=None, names=COLUMN_NAMES
    )

    if survtrace_preprocessing:
        X = _filter_rows_as_survtrace(X)

    # Extract the target events and remove the corresponding columns from the
    # data.
    target_columns = [event_column_name, duration_column_name]
    y, event_labels = _extract_target_events(
        X[target_columns],
        event_column_name,
        duration_column_name,
        censoring_labels,
        events_of_interest=events_of_interest,
        other_event_name=other_event_name,
    )
    X = X.drop(columns=target_columns)

    # Remove columns that should not be used as predictors.
    X = X.drop(columns="Patient ID")

    # TODO: fixme: once informative column names are used, this should be
    # updated accordingly.
    kept_columns = X.columns[~X.columns.str.startswith("Unused")]
    X = X[kept_columns]

    if survtrace_preprocessing:
        X = _preprocess_cols_as_survtrace(X)

    categorical_dtypes = {col: "category" for col in CATEGORICAL_COLUMN_NAMES}
    numerical_dtypes = {col: "float64" for col in NUMERIC_COLUMN_NAMES}

    # Encode missing values with None so that astype will convert missing
    # numerical values to nan and categorical values to pd.NA.

    X = X[CATEGORICAL_COLUMN_NAMES + NUMERIC_COLUMN_NAMES]

    X[CATEGORICAL_COLUMN_NAMES] = X[CATEGORICAL_COLUMN_NAMES].replace("Unknown", None)
    X[NUMERIC_COLUMN_NAMES] = X[NUMERIC_COLUMN_NAMES].replace("Unknown", np.nan)
    X = X.astype({**numerical_dtypes, **categorical_dtypes})

    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    if return_X_y:
        return X, y

    return Bunch(
        X=X,
        y=y,
        event_labels=event_labels,
        original_data=original_data,
    )


def _filter_rows_as_survtrace(X):
    """Filter rows as done in the SurvTRACE paper"""
    mask = (
        # 4 records have N/A cause-specific death classification and are all
        # "Alive".
        (X["SEER cause-specific death classification"] != "N/A not seq 0-59")
        # 282 records with "Survival months" between 1 and 67 months and a mean
        # of 3.2 months. All have a "COD to site recode" with a death cause
        # (non marked as "Alive").
        & (
            X["Reason no cancer-directed surgery"]
            != "Not performed, patient died prior to recommended surgery"
        )
        # The following was part of the original SurvTRACE preprocessing but it
        # is not clear why it is needed: all the values in that column are
        # integers.
        # & (data["Survival months"] != "Unknown")
    )
    return X.loc[mask]


def _preprocess_cols_as_survtrace(X):
    """Replace inplace rare categories to reduce the entropy of the input.

    This function reproduces the preprocessing heuristics from SurvTRACE:
    https://github.com/RyanWangZf/SurvTRACE/blob/main/data/process_seer.py.

    This is achieved by replacing rare categories by either:
    - The category frequency
    - Some placeholder
    - The most frequent category of the longtail, defined by some thresholds.

    Parameters
    ----------
    X : pandas.DataFrame

    Returns
    -------
    X : pandas.DataFrame
    """
    # Make it such that inplace column modifications to not alter the input
    # dataframe
    X = X.copy()

    # Equivalent of OrdinalEncoder on frequencies of "Histologic Type ICD-O-3"
    val_counts = X["Histologic Type ICD-O-3"].value_counts()
    replace_dict = {}
    for idx, count in enumerate(val_counts.values):
        for k in val_counts[val_counts.values == count].index:
            replace_dict[k] = str(idx)

    X["Histologic Type ICD-O-3"].replace(replace_dict, inplace=True)

    # Replace special missing values
    X["ER Status Recode Breast Cancer (1990+)"].replace(
        "Recode not available",
        "Positive",
        inplace=True,
    )
    X["PR Status Recode Breast Cancer (1990+)"].replace(
        "Recode not available",
        "Positive",
        inplace=True,
    )
    X["Summary stage 2000 (1998-2017)"].replace(
        "Unknown/unstaged",
        "Localized",
        inplace=True,
    )
    X["Reason no cancer-directed surgery"].replace(
        "Unknown; death certificate; or autopsy only (2003+)",
        "Surgery performed",
        inplace=True,
    )
    X["Median household income inflation adj to 2019"].replace(
        "Unknown/missing/no match/Not 1990-2018",
        "$75,000+",
        inplace=True,
    )

    min_threshold = {
        "Sequence number": 100,
        "Diagnostic Confirmation": 160,
    }
    for col, min_freq in min_threshold.items():
        val_count = X[col].value_counts()
        low_freq_keys = val_count[val_count < min_freq].index
        replace_dict = {k: low_freq_keys[0] for k in low_freq_keys[1:]}
        X[col].replace(replace_dict, inplace=True)

    return X


def _extract_target_events(
    raw_event_df,
    event_column_name,
    duration_column_name,
    censoring_labels,
    events_of_interest="all",
    other_event_name="Other",
):
    y = raw_event_df.rename(columns={duration_column_name: "duration"})

    if events_of_interest == "all":
        # Encode the event label that corresponds to censoring with 0 and map
        # all the others event labels to integers starting at 1 in
        # lexicographical order.
        events_of_interest = y[event_column_name].unique().tolist()
        for censoring_label in censoring_labels:
            events_of_interest.remove(censoring_label)
        events_of_interest = sorted(events_of_interest)

    event_labels = events_of_interest

    other_event_code = len(events_of_interest) + 1
    event_codes = defaultdict(lambda: other_event_code)

    for censoring_label in censoring_labels:
        event_codes[censoring_label] = 0

    for i, event_name in enumerate(events_of_interest):
        event_codes[event_name] = i + 1

    other_event_code = len(event_codes)
    y["event"] = y[event_column_name].map(event_codes)

    if other_event_code in y["event"].unique():
        event_labels += (other_event_name,)

    return y[["event", "duration"]], np.asarray(event_labels)


class CumulativeOrdinalEncoder(TransformerMixin, BaseEstimator):
    """Ordinal encode all columns as a shared vocabulary.

    Encode categorical values from different columns separately
    e.g. "blue" in column 1 is represented by a different token
    than "blue" in column 2.
    """

    def fit(self, X, y=None):
        del y
        self.ordinal_encoder_ = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,  # We use -1+1=0 as unknown token <UNK>.
        ).fit(X)

        categories = self.ordinal_encoder_.categories_
        vocab_size = [1, *[len(categs) for categs in categories[:-1]]]
        self.cumulated_vocab_size_ = np.cumsum(vocab_size)
        self.vocab_size_ = sum([len(categs) for categs in categories])

        return self

    def transform(self, X):
        X = self.ordinal_encoder_.transform(X)
        X += self.cumulated_vocab_size_
        return X

    def get_feature_names_out(self, input_features=None):
        return self.ordinal_encoder_.get_feature_names_out(input_features)


class FeatureEncoder(TransformerMixin, BaseEstimator):
    """Apply standard scaling and ordinal encoding to the features \
    before tokenizing all categories together.

    Parameters
    ----------
    categorical_columns : iterable of str, default=None
        The categorical column names of the input.
        If set to None, use dtypes to infer.

    numeric_columns : iterable of str, default=None
        The numerical column names of the input.
        If set to None, use dtypes to infer.

    Attributes
    ----------
    categorical_columns_ : iterable of str
        The categorical column names of the input.

    numeric_columns_ : iterable of str
        The numerical column names of the input.

    col_transformer_ : :class:`~sklearn.compose.ColumnTransformer`
        Applies transformers to columns of an array or pandas DataFrame.

    vocab_size_ : ndarray of shape (n_categorical_columns,)
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
    >>> FeatureEncoder().fit_transform(X)
    array([
        [ 0., 2. , -1.22474487],
        [ 1., 3., 0.],
        [ 0., 4., 1.22474487],
    ])

    """

    def __init__(self, categorical_columns=None, numeric_columns=None):
        self.categorical_columns = categorical_columns
        self.numeric_columns = numeric_columns

    def fit(self, X, y=None):
        del y
        X = self._check_num_categorical_columns(X)

        encoder = make_pipeline(
            SimpleImputer(strategy="most_frequent"),
            CumulativeOrdinalEncoder(),
        )

        self.col_transformer_ = make_column_transformer(
            (encoder, self.categorical_columns_),
            (StandardScaler(), self.numeric_columns_),
            remainder="drop",
            verbose_feature_names_out=False,
        )
        self.col_transformer_.set_output(transform="pandas")
        self.col_transformer_.fit(X)

        return self

    def transform(self, X, y=None):
        del y
        check_is_fitted(self, "col_transformer_")
        return self.col_transformer_.transform(X)

    @property
    def vocab_size_(self):
        check_is_fitted(self, "col_transformer_")
        if len(self.categorical_columns_) == 0:
            return 0
        return self.col_transformer_.transformers_[0][1][1].vocab_size_

    def _check_num_categorical_columns(self, X):
        if not hasattr(X, "__dataframe__"):
            raise TypeError(f"X must be a dataframe, got {type(X)}.")

        X = X.copy()  # needed since we make inplace changes to the dataframe.

        if self.numeric_columns is None:
            int_columns = X.select_dtypes("int").columns
            if int_columns.shape[0] > 0:
                raise ValueError(
                    "Integer dtypes are ambiguous for numeric "
                    f"columns {int_columns!r}.\n"
                    "Please convert them to float dtypes or set "
                    "'numeric_columns'."
                )
            self.numeric_columns_ = X.select_dtypes("float").columns.tolist()
        else:
            self.numeric_columns_ = np.atleast_1d(self.numeric_columns).tolist()
        X[self.numeric_columns_] = X[self.numeric_columns_].astype("float")

        if self.categorical_columns is None:
            object_columns = X.select_dtypes(["bool", "object", "string"]).columns
            if object_columns.shape[0] > 0:
                raise ValueError(
                    "Object, boolean and string dtypes are ambiguous for categorical "
                    f"columns {object_columns!r}.\n"
                    "Please convert them to category dtypes or set "
                    "'categorical_columns'."
                )
            self.categorical_columns_ = X.select_dtypes(["category"]).columns.tolist()
        else:
            self.categorical_columns_ = np.atleast_1d(self.categorical_columns).tolist()
        X[self.categorical_columns_] = X[self.categorical_columns_].astype("category")

        return X
