import numpy as np
import pandas as pd
from pathlib import Path
import json

PATH_PREDICTIONS = Path("preds/")
DATASET_NAME = "competing_weibull"


def load_predictions(path_predictions: Path, dataset_name: str) -> dict:
    """
    Charge les prédictions stockées en fichiers Parquet dans un dictionnaire.

    Args:
        path_predictions (Path): Chemin du dossier racine contenant les prédictions.
        dataset_name (str): Nom du dataset.

    Returns:
        dict: Dictionnaire structuré {nom_du_modele: {nom_prediction: np.array}}
    """
    dataset_path = path_predictions / dataset_name
    predictions_dict = {}

    for model_path in dataset_path.iterdir():  # Itère sur les modèles
        print(model_path)
        if not model_path.is_dir():
            continue  # Ignore les fichiers qui ne sont pas des dossiers de modèle

        model_name = model_path.name
        predictions_dict[model_name] = {}
        for seed_path in model_path.iterdir():  # Itère sur les seeds
            if not seed_path.is_dir():
                continue
            seed_name = int(seed_path.name[-1])
            predictions_dict[model_name][seed_name] = {}
            for (
                pred_type_path
            ) in seed_path.iterdir():  # Itère sur les types de prédictions
                if not pred_type_path.is_dir():
                    if pred_type_path.suffix == ".parquet":
                        if str(pred_type_path).split("/")[-1] == "times.parquet":
                            times = pd.read_parquet(pred_type_path).values.flatten()
                            predictions_dict[model_name][seed_name]["times"] = times
                        elif str(pred_type_path).split("/")[-1] == "index.parquet":
                            indexes = pd.read_parquet(pred_type_path)
                            predictions_dict[model_name][seed_name]["y_conf_index"] = (
                                indexes.loc["y_conf", :].dropna().astype(int).values
                            )
                            predictions_dict[model_name][seed_name]["y_train_index"] = (
                                indexes.loc["y_train", :].dropna().astype(int).values
                            )
                            predictions_dict[model_name][seed_name]["y_test_index"] = (
                                indexes.loc["y_test", :].dropna().astype(int).values
                            )
                    continue

                pred_type = pred_type_path.name
                event_list = []

                for event_file in sorted(
                    pred_type_path.glob("event_*.parquet")
                ):  # Charge les événements
                    df = pd.read_parquet(event_file)
                    event_list.append(df.values)  # Convertit en numpy array

                if event_list:
                    predictions_dict[model_name][seed_name][pred_type] = np.stack(
                        event_list, axis=1
                    )  # Reconstruction du 3D array

    return predictions_dict


def load_metrics(path_predictions: Path, dataset_name: str) -> dict:
    """
    Charge les prédictions stockées en fichiers Parquet dans un dictionnaire.

    Args:
        path_predictions (Path): Chemin du dossier racine contenant les prédictions.
        dataset_name (str): Nom du dataset.

    Returns:
        dict: Dictionnaire structuré {nom_du_modele: {nom_prediction: np.array}}
    """
    dataset_path = path_predictions / dataset_name
    predictions_dict = {}

    for model_path in dataset_path.iterdir():  # Itère sur les modèles
        print(model_path)
        if not model_path.is_dir():
            continue  # Ignore les fichiers qui ne sont pas des dossiers de modèle

        model_name = model_path.name
        predictions_dict[model_name] = {}
        for recalibration in model_path.iterdir():  # Itère sur les recalibrations
            if not recalibration.is_dir():
                continue
            is_recalibration = recalibration.name
            predictions_dict[model_name][is_recalibration] = {}
            for seed_path in recalibration.iterdir():  # Itère sur les seeds
                if not seed_path.is_dir():
                    continue
                seed_name = int(seed_path.name[-1])
                predictions_dict[model_name][is_recalibration][seed_name] = {}
                # load metrics in a json
                path_file_agg = seed_path / "metrics.json"
                if path_file_agg.exists():
                    with open(path_file_agg, "r") as f:
                        metrics = json.load(f)
                    predictions_dict[model_name][is_recalibration][seed_name] = metrics
                else:
                    print(f"File {path_file_agg} does not exist.")
                    continue
                # load predictions for visualization
                path_dir_pred = seed_path / "vizualization_mean_predictions.csv"
                if path_dir_pred.exists():
                    prediction_whole_train = pd.read_parquet(path_dir_pred)
                    predictions_dict[model_name][is_recalibration][seed_name][
                        "prediction_whole_train"
                    ] = prediction_whole_train.values
                else:
                    print(f"File {path_dir_pred} does not exist.")
                    continue
    return predictions_dict
