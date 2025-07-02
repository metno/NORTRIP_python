from functions import longwave_in_radiation_func


def test_longwave_in_radiation_func():
    """Test the longwave radiation function."""
    TC = 10.0  # 10°C
    RH = 60.0  # 60%
    n_c = 0.3
    P = 101325.0  # Pa

    RL_in = longwave_in_radiation_func(TC, RH, n_c, P)

    assert isinstance(RL_in, float)
    assert RL_in > 0
    assert RL_in < 500  # Reasonable upper bound for longwave radiation
