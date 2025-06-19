from src.config_classes.model_flags import model_flags


def test_model_flags_initialization():
    # Test that the dataclass can be initialized without errors
    flags = model_flags()
    assert flags is not None
