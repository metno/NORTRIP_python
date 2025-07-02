import numpy as np
from src.functions.mass_balance_func import mass_balance_func


def test_mass_balance_func():
    """Test temporal mass balance calculation."""

    # Test case 1: Normal case where P < R * 1e8
    M_0 = 10.0  # Initial mass
    P = 5.0  # Production term
    R = 0.1  # Removal rate
    dt = 1.0  # Time step

    result = mass_balance_func(M_0, P, R, dt)

    # Calculate expected result manually
    expected = P / R * (1 - np.exp(-R * dt)) + M_0 * np.exp(-R * dt)
    assert abs(result - expected) < 1e-10

    # Result should be positive
    assert result > 0

    # Test case 2: Large production term (P >= R * 1e8)
    M_0 = 10.0
    P = 1e10  # Very large production
    R = 0.1
    dt = 1.0

    result = mass_balance_func(M_0, P, R, dt)

    # Should use linear approximation: M = M_0 + P * dt
    expected = M_0 + P * dt
    assert abs(result - expected) < 1e-10

    # Test case 3: Zero production
    M_0 = 20.0
    P = 0.0
    R = 0.2
    dt = 2.0

    result = mass_balance_func(M_0, P, R, dt)

    # Should decay exponentially: M = M_0 * exp(-R * dt)
    expected = M_0 * np.exp(-R * dt)
    assert abs(result - expected) < 1e-10

    # Test case 4: Zero removal rate
    M_0 = 15.0
    P = 3.0
    R = 0.0
    dt = 1.5

    result = mass_balance_func(M_0, P, R, dt)

    # Should be linear accumulation: M = M_0 + P * dt
    expected = M_0 + P * dt
    assert abs(result - expected) < 1e-10


def test_mass_balance_func_conservation():
    """Test mass conservation properties."""

    # Test that mass is conserved over multiple time steps
    M_0 = 100.0
    P = 10.0
    R = 0.05
    dt = 0.1

    # Single large time step
    result_single = mass_balance_func(M_0, P, R, 10 * dt)

    # Multiple smaller time steps
    M = M_0
    for i in range(10):
        M = mass_balance_func(M, P, R, dt)
    result_multiple = M

    # Results should be approximately equal
    assert abs(result_single - result_multiple) < 0.01


def test_mass_balance_func_steady_state():
    """Test approach to steady state."""

    P = 5.0
    R = 0.1
    M_0 = 0.0
    dt = 1.0

    # Steady state should be P/R
    expected_steady_state = P / R

    # Run for many time steps
    M = M_0
    for i in range(100):
        M = mass_balance_func(M, P, R, dt)

    # Should approach steady state
    assert abs(M - expected_steady_state) < 0.1
