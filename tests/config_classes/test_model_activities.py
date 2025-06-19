from src.config_classes.model_activities import model_activities


def test_model_activities_initialization():
    # Test that the dataclass can be initialized without errors
    activities = model_activities()
    assert activities is not None
