from src.input_classes.input_values_initial import input_values_initial


def test_input_values_initial_initialization():
    # Test that the dataclass can be initialized without errors
    initial_values_data = input_values_initial()
    assert initial_values_data is not None
