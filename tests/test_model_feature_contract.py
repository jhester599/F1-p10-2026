from config import FEATURE_COLS, MODEL_FEATURES


def test_global_feature_list_has_no_duplicates() -> None:
    assert len(FEATURE_COLS) == len(set(FEATURE_COLS))


def test_each_model_feature_subset_is_known_and_non_empty() -> None:
    global_features = set(FEATURE_COLS)

    for model_name, model_features in MODEL_FEATURES.items():
        assert model_features, f"{model_name} has no features"
        assert len(model_features) == len(set(model_features)), f"{model_name} has duplicate features"
        assert set(model_features) <= global_features, f"{model_name} references unknown features"
