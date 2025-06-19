from src.input_classes.input_initial import input_initial


def test_input_initial_initialization():
    # Test that the dataclass can be initialized without errors
    initial_data = input_initial()
    assert initial_data is not None
