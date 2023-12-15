"""Helper to reproduce the SEER dataset transformations as performed in SurvTRACE.

See: https://github.com/RyanWangZf/SurvTRACE/blob/main/data/process_seer.py
"""

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets._base import Bunch  # XXX: use a dataclass instead?

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

# "Undefined x" names are placeholders, these columns are not used by SurvTRACE.
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
            The two columns are named "event" and "duration". The "event"
            columns holds integer idenfiers of event of interest or 0 for
            censoring. The meaning of event integer codes is defined by the
            position in the event_labels list. The "duration" columns holds a
            numerical value for the event free duration expressed in months. #
            TODO: document what t0 mean.

        event_labels : list of str
            The labels of the events.

        original_data : pandas.DataFrame of shape (n_samples, 29)
            The original data.
    """
    msg = (
        f"The SEER extracted file doesn't exist at {input_path}."
        "See the installation guide at https://soda-inria.github.io/hazardous/xxx."
    )
    if not Path(input_path).exists():
        raise FileNotFoundError(msg)

    # XXX: specify dtypes for all columns at load time insted of retyping a
    # posteriori?
    # In particular this should help remove a pandas warning for column 22.
    original_data = data = pd.read_csv(
        input_path, sep="\t", header=None, names=COLUMN_NAMES
    )

    if survtrace_preprocessing:
        data = _filter_rows_as_survtrace(data)

    # Extract the target events and remove the corresponding columns from the
    # data.
    target_columns = [event_column_name, duration_column_name]
    target, event_labels = _extract_target_events(
        data[target_columns],
        event_column_name,
        duration_column_name,
        censoring_labels,
        events_of_interest=events_of_interest,
        other_event_name=other_event_name,
    )
    data = data.drop(columns=target_columns)

    # Remove columns that should not be used as predictors.
    data = data.drop(columns="Patient ID")

    # TODO: fixme: once informative column names are used, this should be
    # updated accordingly.
    kept_columns = data.columns[~data.columns.str.startswith("Unused")]
    data = data[kept_columns]

    if survtrace_preprocessing:
        data = _preprocess_cols_as_survtrace(data)

    categorical_dtypes = {col: "category" for col in CATEGORICAL_COLUMN_NAMES}

    # There are no decimal values in the numerical columns so let's use int64.
    numerical_dtypes = {col: "int64" for col in NUMERIC_COLUMN_NAMES}

    # Encode missing values with None so that astype will convert missing
    # numerical values to nan and categorical values to pd.NA.
    data.replace("Unknown", None, inplace=True)
    data = data.astype({**numerical_dtypes, **categorical_dtypes})

    if return_X_y:
        return data, target

    return Bunch(
        data=data,
        target=target,
        event_labels=event_labels,
        original_data=original_data,
    )


def _filter_rows_as_survtrace(data):
    """Filter rows as done in the SurvTRACE paper"""
    mask = (
        # 4 records have N/A cause-specific death classification and are all
        # "Alive".
        (data["SEER cause-specific death classification"] != "N/A not seq 0-59")
        # 282 records with "Survival months" between 1 and 67 months and a mean
        # of 3.2 months. All have a "COD to site recode" with a death cause
        # (non marked as "Alive").
        & (
            data["Reason no cancer-directed surgery"]
            != "Not performed, patient died prior to recommended surgery"
        )
        # The following was part of the original SurvTRACE preprocessing but it
        # is not clear why it is needed: all the values in that column are
        # integers.
        # & (data["Survival months"] != "Unknown")
    )
    return data.loc[mask]


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
    target = raw_event_df.rename(columns={duration_column_name: "duration"})

    if events_of_interest == "all":
        # Encode the event label that corresponds to censoring with 0 and map
        # all the others event labels to integers starting at 1 in
        # lexicographical order.
        events_of_interest = target[event_column_name].unique().tolist()
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
    target["event"] = target[event_column_name].map(event_codes)

    if other_event_code in target["event"].unique():
        event_labels += (other_event_name,)

    return target[["event", "duration"]], np.asarray(event_labels)
