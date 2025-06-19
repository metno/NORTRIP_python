from src.config_classes.model_file_paths import model_file_paths


def test_model_file_paths_initialization():
    # Test that the dataclass can be initialized without errors
    file_paths = model_file_paths()
    assert file_paths is not None
