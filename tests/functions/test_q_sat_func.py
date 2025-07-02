from functions import q_sat_func


def test_q_sat_func():
    """Test the saturation vapor pressure function."""
    TC = 20.0  # 20°C
    P = 101325.0  # Pa

    esat, qsat, s = q_sat_func(TC, P)

    assert isinstance(esat, float)
    assert isinstance(qsat, float)
    assert isinstance(s, float)
    assert esat > 0  # Should be positive
    assert qsat > 0  # Should be positive
    assert s > 0  # Should be positive
