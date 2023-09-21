"""Helper to reproduce the SEER dataset transformations as performed in SurvTRACE.

See: https://github.com/RyanWangZf/SurvTRACE/blob/main/data/process_seer.py
"""
import pandas as pd

CAT_COLS = [
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

NUM_COLS = [
    "Regional nodes examined (1988+)",
    "CS tumor size (2004-2015)",
    "Total number of benign/borderline tumors for patient",
    "Total number of in situ/malignant tumors for patient",
]

COLS_MAPPING = {
    0: "Patient ID",
    1: "Sex",
    2: "Year of diagnosis",
    3: "Site recode ICD-0-3/WHO 2008",
    4: "COD to site recode",
    5: "Race recode (W, B, AI, API)",
    6: "Histologic Type ICD-O-3",
    7: "ICD-O-3 Hist/behav, malignant",
    8: "Laterality",
    9: "Sequence number",
    10: "Vital status recode (study cutoff used)",
    11: "ER Status Recode Breast Cancer (1990+)",
    12: "PR Status Recode Breast Cancer (1990+)",
    13: "Regional nodes examined (1988+)",
    14: "xxx",
    15: "Summary stage 2000 (1998-2017)",
    16: "xxx",
    17: "Reason no cancer-directed surgery",
    18: "CS tumor size (2004-2015)",
    19: "First malignant primary indicator",
    20: "Diagnostic Confirmation",
    21: "Total number of benign/borderline tumors for patient",
    22: "Total number of in situ/malignant tumors for patient",
    23: "Median household income inflation adj to 2019",
    24: "RX Summ--Surg Prim Site (1998+)",
    25: "SEER other cause of death classification",
    26: "SEER cause-specific death classification",
    27: "Survival months",
    28: "xxx",
}


def preprocess_X_y(input_path, rename=True):
    if rename:
        df = pd.read_csv(input_path, sep="\t", header=None)
        df = add_column_names(df)
    else:
        df = pd.read_csv(input_path)

    mask = (
        (df["SEER cause-specific death classification"] != "N/A not seq 0-59")
        & (
            df["Reason no cancer-directed surgery"]
            != "Not performed, patient died prior to recommended surgery"
        )
        & (df["Survival months"] != "Unknown")
    )
    df = df.loc[mask]

    df_features = preprocess_features(df)
    df_events = preprocess_events(df)

    return df_features, df_events


def add_column_names(df):
    """Name the columns of the SEER dataset, when missing from the extraction."""
    df.rename(COLS_MAPPING, axis=1, inplace=True)
    return df


def preprocess_features(df):
    df = df.drop(columns="Patient ID")

    # Equivalent of OrdinalEncoder on frequencies of "Histologic Type ICD-O-3"
    val_counts = df["Histologic Type ICD-O-3"].value_counts()
    replace_dict = {}
    for idx, count in enumerate(val_counts.values):
        for k in val_counts[val_counts.values == count].index:
            replace_dict[k] = str(idx)

    df["Histologic Type ICD-O-3"].replace(replace_dict, inplace=True)

    # Replace special missing values
    df["ER Status Recode Breast Cancer (1990+)"].replace(
        "Recode not available",
        "Positive",
        inplace=True,
    )
    df["PR Status Recode Breast Cancer (1990+)"].replace(
        "Recode not available",
        "Positive",
        inplace=True,
    )
    df["Summary stage 2000 (1998-2017)"].replace(
        "Unknown/unstaged",
        "Localized",
        inplace=True,
    )
    df["Reason no cancer-directed surgery"].replace(
        "Unknown; death certificate; or autopsy only (2003+)",
        "Surgery performed",
        inplace=True,
    )
    df["Median household income inflation adj to 2019"].replace(
        "Unknown/missing/no match/Not 1990-2018",
        "$75,000+",
        inplace=True,
    )

    df.replace("Unknown", None, inplace=True)

    # TODO: specialize this logic between training and inference.
    # During inference, we should fetch the trained object.
    min_threshold = {
        "Sequence number": 100,
        "Diagnostic Confirmation": 160,
    }
    for col, min_freq in min_threshold.items():
        val_count = df[col].value_counts()
        low_freq_keys = val_count[val_count < min_freq].index
        replace_dict = {k: low_freq_keys[0] for k in low_freq_keys[1:]}
        df[col].replace(replace_dict, inplace=True)

    return df


def preprocess_events(df):
    df = df.rename(columns={"Survival months": "duration"})

    target_encoded = {
        "Breast": 1,
        "Diseases of Heart": 2,
    }
    df["event"] = df["COD to site recode"].map(target_encoded).fillna(0)

    return df[["event", "duration"]]
