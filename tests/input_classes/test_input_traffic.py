from src.input_classes import input_traffic


def test_input_traffic_initialization():
    # Test that the dataclass can be initialized without errors
    traffic_data = input_traffic()
    assert traffic_data is not None
