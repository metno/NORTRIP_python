from src.input_classes import input_meteorology


def test_input_meteorology_initialization():
    # Test that the dataclass can be initialized without errors
    meteorology_data = input_meteorology()
    assert meteorology_data is not None
