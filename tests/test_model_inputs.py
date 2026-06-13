import numpy as np
import pandas as pd
import pytest

from config import FEATURE_COLS
from src import models


def test_model_features_for_known_model_uses_configured_subset() -> None:
    assert models._model_features("ridge") == models.MODEL_FEATURES["ridge"]


def test_model_features_for_unknown_model_uses_full_feature_list() -> None:
    assert models._model_features("custom_model") == FEATURE_COLS


def test_feature_matrix_for_model_raises_clear_error_for_missing_training_column() -> None:
    df = pd.DataFrame({"grid_position": [10.0]})

    with pytest.raises(KeyError, match="Missing required model feature columns"):
        models._feature_matrix_for_model(df, "ridge")


def test_prepare_race_features_for_prediction_backfills_missing_columns() -> None:
    race_features = pd.DataFrame(
        {
            "driver_id": ["a"],
            "constructor_id": ["team"],
            "grid_position": [10.0],
        }
    )

    prepared = models._prepare_race_features_for_prediction(race_features)

    assert set(FEATURE_COLS) <= set(prepared.columns)
    assert prepared.loc[0, "grid_position"] == 10.0
    missing_cols = [col for col in FEATURE_COLS if col != "grid_position"]
    assert np.allclose(prepared.loc[0, missing_cols].to_numpy(dtype=float), 0.0)
    assert list(race_features.columns) == ["driver_id", "constructor_id", "grid_position"]
