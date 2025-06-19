from src.input_classes import input_metadata


def test_input_metadata_initialization():
    # Test that the dataclass can be initialized without errors
    metadata = input_metadata()
    assert metadata is not None
