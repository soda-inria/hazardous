"""Helper to reproduce the SEER dataset transformations as performed in SurvTRACE.

See: https://github.com/RyanWangZf/SurvTRACE/blob/main/data/process_seer.py
"""
from pathlib import Path

import pandas as pd

CATEGORICAL_COLUMN_NAMES = [
    "Sex",
    "Year of diagnosis",
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
    "Undefined 1",
    "Summary stage 2000 (1998-2017)",
    "Undefined 2",
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
    "Undefined 3",
]


def load_seer(input_path, survtrace_preprocessing=False):
    """Load the seer dataset and optionally apply the same preprocessing \
        as done in SurvTRACE.

    The file is expected to be a txt file.

    Parameters
    ----------
    input_path : str or file_path
        The path of the txt file.

    survtrace_preprocessing : bool, default=False
        If set to True, apply the preprocessing steps used in SurvTRACE to
        ensure reproducibility.

    Returns
    -------
    X : pandas.DataFrame of shape (n_samples, n_features)
        The dataframe of features.

    y : pandas.DataFrame of shape (n_samples, 2)
        The dataframe of targets, with columns 'event' and 'duration'.
        The event 1 is 'Breast Cancer', and the event 2 is 'Disease of Heart'.
        The event 0 indicates censoring.
    """
    msg = (
        f"The SEER extracted file doesn't exist at {input_path}."
        "See the installation guide at https://soda-inria.github.io/hazardous/xxx."
    )
    if not Path(input_path).exists():
        raise FileNotFoundError(msg)

    X = pd.read_csv(input_path, sep="\t", header=None, names=COLUMN_NAMES)

    if survtrace_preprocessing:
        X = preprocess_features_as_survtrace(X)

    y = preprocess_events(X)

    categorical_dtypes = {col: "category" for col in CATEGORICAL_COLUMN_NAMES}
    numerical_dtypes = {col: "float64" for col in NUMERIC_COLUMN_NAMES}
    X = X.astype({**numerical_dtypes, **categorical_dtypes}).drop(
        columns=["COD to site recode", "Survival months"]
    )

    return X, y


def preprocess_features_as_survtrace(X):
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
    X = X.drop(columns="Patient ID")

    mask = (
        (X["SEER cause-specific death classification"] != "N/A not seq 0-59")
        & (
            X["Reason no cancer-directed surgery"]
            != "Not performed, patient died prior to recommended surgery"
        )
        & (X["Survival months"] != "Unknown")
    )
    X = X.loc[mask]

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

    X.replace("Unknown", None, inplace=True)

    min_threshold = {
        "Sequence number": 100,
        "Diagnostic Confirmation": 160,
    }
    for col, min_freq in min_threshold.items():
        val_count = X[col].value_counts()
        low_freq_keys = val_count[val_count < min_freq].index
        replace_dict = {k: low_freq_keys[0] for k in low_freq_keys[1:]}
        X[col].replace(replace_dict, inplace=True)

    column_names = X.columns[~X.columns.str.contains("Undefined")]

    return X[column_names]


def preprocess_events(X):
    """Ordinal encode the events and rename event columns.

    Returns a copy of X.

    Parameters
    ----------
    X : pandas.DataFrame

    Returns
    -------
    X : pandas.DataFrame of shape (n_samples, 2)
    """
    X = X.rename(columns={"Survival months": "duration"})

    target_encoded = {
        "Breast": 1,
        "Diseases of Heart": 2,
    }
    X["event"] = X["COD to site recode"].map(target_encoded).fillna(0)

    return X[["event", "duration"]]
