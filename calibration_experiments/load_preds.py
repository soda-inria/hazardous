import numpy as np
import pandas as pd
from pathlib import Path

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
        if not model_path.is_dir():
            continue  # Ignore les fichiers qui ne sont pas des dossiers de modèle

        model_name = model_path.name
        predictions_dict[model_name] = {}

        for (
            pred_type_path
        ) in model_path.iterdir():  # Itère sur les types de prédictions
            if not pred_type_path.is_dir():
                if pred_type_path.suffix == ".parquet":
                    times = pd.read_parquet(pred_type_path).values.flatten()
                    predictions_dict[model_name]["times"] = times
                continue

            pred_type = pred_type_path.name
            event_list = []

            for event_file in sorted(
                pred_type_path.glob("event_*.parquet")
            ):  # Charge les événements
                df = pd.read_parquet(event_file)
                event_list.append(df.values)  # Convertit en numpy array

            if event_list:
                predictions_dict[model_name][pred_type] = np.stack(
                    event_list, axis=1
                )  # Reconstruction du 3D array

    return predictions_dict
