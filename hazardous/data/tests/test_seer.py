from pathlib import Path

import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal

from hazardous.data._seer import (
    CATEGORICAL_COLUMN_NAMES,
    NUMERIC_COLUMN_NAMES,
    load_seer,
)

DIR_SAMPLES = Path(__file__).parent
DIR_DATA = Path(__file__).parent.parent


def test_load_seer_from_fake_sample_file():
    input_path = DIR_SAMPLES / "fake_seer_sample.txt"

    seer_dataset = load_seer(input_path)

    event_labels = seer_dataset.event_labels
    assert list(event_labels) == ["Breast", "Diseases of Heart", "Other"]

    X = seer_dataset.data
    assert X.shape == (3, 23)

    expected_y = pd.DataFrame(
        dict(
            # 0 is the censoring marker, 1 is "Breast", 3 is "Other"
            event=[3, 0, 1],
            duration=[7, 81, 28],
        )
    )
    y = seer_dataset.target
    assert_frame_equal(y, expected_y)

    categorical_column_names = X.select_dtypes("category").columns
    assert sorted(categorical_column_names) == sorted(CATEGORICAL_COLUMN_NAMES)

    numeric_column_names = X.select_dtypes("number").columns
    assert sorted(numeric_column_names) == sorted(NUMERIC_COLUMN_NAMES)


raw_seer_path = DIR_DATA / "seer_cancer_cardio_raw_data.txt"


@pytest.mark.skipif(
    not raw_seer_path.exists(), reason=f"{raw_seer_path} doesn't exist."
)
def test_load_seer():
    X, y = load_seer(raw_seer_path, survtrace_preprocessing=True, return_X_y=True)

    assert X.shape[0] == 476_746
    assert X.shape[0] == y.shape[0]

    assert sorted(y.columns) == ["duration", "event"]
    assert dict(y["event"].value_counts()) == {
        0: 298168,  # Alive
        1: 87495,  # Breast
        2: 21549,  # Diseases of Heart
        3: 69534,  # Other
    }
    assert y["duration"].mean().round(2) == 67.41

    categorical_column_names = X.select_dtypes("category").columns
    assert sorted(categorical_column_names) == sorted(CATEGORICAL_COLUMN_NAMES)

    numeric_column_names = X.select_dtypes("number").columns
    assert sorted(numeric_column_names) == sorted(NUMERIC_COLUMN_NAMES)

    expected_nunique = {
        "Sex": 2,
        "Year of diagnosis": 11,
        "Race recode (W, B, AI, API)": 4,
        "Histologic Type ICD-O-3": 75,
        "Laterality": 5,
        "Sequence number": 7,
        "ER Status Recode Breast Cancer (1990+)": 3,
        "PR Status Recode Breast Cancer (1990+)": 3,
        "Summary stage 2000 (1998-2017)": 3,
        "RX Summ--Surg Prim Site (1998+)": 7,
        "Reason no cancer-directed surgery": 6,
        "First malignant primary indicator": 2,
        "Diagnostic Confirmation": 6,
        "Median household income inflation adj to 2019": 10,
    }
    assert X[CATEGORICAL_COLUMN_NAMES].nunique().to_dict() == expected_nunique

    expected_mean = {
        "Regional nodes examined (1988+)": 9.02,
        "CS tumor size (2004-2015)": 99.51,
        "Total number of benign/borderline tumors for patient": 0.01,
        "Total number of in situ/malignant tumors for patient": 1.38,
    }
    assert X[NUMERIC_COLUMN_NAMES].mean().round(2).to_dict() == expected_mean

    expected_max = {
        "Regional nodes examined (1988+)": 99.0,
        "CS tumor size (2004-2015)": 999.0,
        "Total number of benign/borderline tumors for patient": 5.0,
        "Total number of in situ/malignant tumors for patient": 20.0,
    }
    assert X[NUMERIC_COLUMN_NAMES].max().to_dict() == expected_max

    expected_min = {
        "Regional nodes examined (1988+)": 0.0,
        "CS tumor size (2004-2015)": 0.0,
        "Total number of benign/borderline tumors for patient": 0.0,
        "Total number of in situ/malignant tumors for patient": 1.0,
    }
    assert X[NUMERIC_COLUMN_NAMES].min().to_dict() == expected_min

    expected_isna_sum = {
        "Sex": 0,
        "Year of diagnosis": 0,
        "Site recode ICD-0-3/WHO 2008": 0,
        "Race recode (W, B, AI, API)": 2116,
        "Histologic Type ICD-O-3": 0,
        "ICD-O-3 Hist/behav, malignant": 0,
        "Laterality": 0,
        "Sequence number": 0,
        "Vital status recode (study cutoff used)": 0,
        "ER Status Recode Breast Cancer (1990+)": 0,
        "PR Status Recode Breast Cancer (1990+)": 0,
        "Regional nodes examined (1988+)": 0,
        "Summary stage 2000 (1998-2017)": 0,
        "Reason no cancer-directed surgery": 0,
        "CS tumor size (2004-2015)": 0,
        "First malignant primary indicator": 0,
        "Diagnostic Confirmation": 1561,
        "Total number of benign/borderline tumors for patient": 0,
        "Total number of in situ/malignant tumors for patient": 0,
        "Median household income inflation adj to 2019": 0,
        "RX Summ--Surg Prim Site (1998+)": 0,
        "SEER other cause of death classification": 0,
        "SEER cause-specific death classification": 0,
    }
    assert X.isna().sum().to_dict() == expected_isna_sum

    object_column_names = X.select_dtypes("object").columns
    expected_object_column_names = [
        "Site recode ICD-0-3/WHO 2008",
        "ICD-O-3 Hist/behav, malignant",
        "Vital status recode (study cutoff used)",
        "SEER other cause of death classification",
        "SEER cause-specific death classification",
    ]
    assert_array_equal(object_column_names, expected_object_column_names)
