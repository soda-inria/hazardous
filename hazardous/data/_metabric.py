import pandas as pd
from sklearn.datasets._base import Bunch
from skrub import TableVectorizer

COLUMNS_COVARIATES = [
    "Cohort",
    "Age_At_Diagnosis",
    "Breast_Tumour_Laterality",
    "year_diag",
    "NPI",
    "ER_Status",
    "Inferred_Menopausal_State",
    "Lymph_Nodes_Positive",
    "Breast_Surgery",
    "CT",
    "HT",
    "RT",
    "Grade",
    "Size",
    "Histological_Type",
    "Stage",
]

TARGET_COLUMNS = ["Death", "T", "TLR", "TDR", "LR", "DR"]


def load_metabric(
    input_path=None,
    return_X_y=False,
    event_column_name="event",
    duration_column_name="duration",
    kept_columns=COLUMNS_COVARIATES,
    target_columns=TARGET_COLUMNS,
):
    """ """
    if input_path is None:
        df = pd.read_csv(
            "https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-019-1007-8/MediaObjects/41586_2019_1007_MOESM7_ESM.txt",
            sep="\t",
        )
        df["year_diag"] = df["Date.Of.Diagnosis"].str[:4].astype(int)
        df.columns = df.columns.str.replace(".", "_")
        df.columns = df.columns.str.replace(" ", "_")
    # Extract the target events and remove the corresponding columns from the
    # data.

    vectorizer = TableVectorizer()
    vect_df = vectorizer.fit_transform(
        df.dropna(subset=target_columns, how="any")[kept_columns + target_columns]
    )
    kept_columns = vect_df.columns.difference(target_columns)
    X = vect_df[kept_columns]
    X = X.fillna(X.mean())

    y_ = vect_df[target_columns]
    y_["event"] = (y_["LR"] + y_["DR"]) > 0
    y_["duration"] = y_[["TLR", "TDR"]].min(axis=1)
    mask_competing_event = (y_["T"] <= y_["duration"]) & (y_["Death"] == 1)
    y_.loc[mask_competing_event, "event"] = 2
    y_.loc[mask_competing_event, "duration"] = y_.loc[
        (y_["T"] <= y_["duration"]) & (y_["Death"]), "T"
    ]

    # relapse with the CR event of death
    y = y_[["event", "duration"]].copy()
    y["event"] = y["event"].astype(int)
    X = X.astype(float)

    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    if return_X_y:
        return X, y

    return Bunch(
        X=X,
        y=y,
    )
