from src.config_classes.model_parameters import model_parameters


def test_model_parameters_initialization():
    # Test that the dataclass can be initialized without errors
    params = model_parameters()
    assert params is not None
