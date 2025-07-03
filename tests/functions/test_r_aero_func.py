import numpy as np
from src.functions.r_aero_func import r_aero_func


def test_basic_functionality():
    """Test basic aerodynamic resistance calculation."""
    # Test parameters
    FF = 5.0  # Wind speed m/s
    z_FF = 10.0  # Wind measurement height m
    z_T = 2.0  # Temperature measurement height m
    z0 = 0.01  # Roughness length for momentum m
    z0t = 0.001  # Roughness length for heat m
    V_veh = np.array([80.0, 60.0, 40.0])  # Vehicle velocities km/h
    N_v = np.array([100.0, 50.0, 20.0])  # Number of vehicles
    num_veh = 3
    a_traffic = np.array([0.1, 0.08, 0.05])  # Traffic coefficients

    r_aero = r_aero_func(FF, z_FF, z_T, z0, z0t, V_veh, N_v, num_veh, a_traffic)

    assert isinstance(r_aero, float)
    assert r_aero > 0
    # Check reasonable range for aerodynamic resistance
    assert 1 < r_aero < 1000


def test_low_wind_speed():
    """Test behavior with very low wind speed (should use minimum of 0.2 m/s)."""
    FF = 0.01  # Very low wind speed
    z_FF = 10.0
    z_T = 2.0
    z0 = 0.01
    z0t = 0.001
    V_veh = np.array([80.0])
    N_v = np.array([100.0])
    num_veh = 1
    a_traffic = np.array([0.1])

    r_aero = r_aero_func(FF, z_FF, z_T, z0, z0t, V_veh, N_v, num_veh, a_traffic)

    # Should still produce valid result due to minimum wind speed
    assert r_aero > 0
    assert np.isfinite(r_aero)


def test_no_traffic():
    """Test with no traffic (only wind contribution)."""
    FF = 5.0
    z_FF = 10.0
    z_T = 2.0
    z0 = 0.01
    z0t = 0.001
    V_veh = np.array([80.0, 60.0])
    N_v = np.array([0.0, 0.0])  # No vehicles
    num_veh = 2
    a_traffic = np.array([0.1, 0.08])

    r_aero = r_aero_func(FF, z_FF, z_T, z0, z0t, V_veh, N_v, num_veh, a_traffic)

    # Calculate expected value with only wind contribution
    kappa = 0.4
    # Use max(FF, 0.2) to match implementation
    inv_r_wind = max(FF, 0.2) * kappa**2 / (np.log(z_FF / z0) * np.log(z_T / z0t))
    # Account for minimum traffic contribution
    inv_r_traffic = 1e-6  # Minimum value from implementation
    expected_r_aero = 1 / (inv_r_wind + inv_r_traffic)

    assert np.isclose(r_aero, expected_r_aero, rtol=1e-10)


def test_zero_vehicle_speed():
    """Test with zero vehicle speeds."""
    FF = 5.0
    z_FF = 10.0
    z_T = 2.0
    z0 = 0.01
    z0t = 0.001
    V_veh = np.array([0.0, 0.0])  # Zero speeds
    N_v = np.array([100.0, 50.0])
    num_veh = 2
    a_traffic = np.array([0.1, 0.08])

    r_aero = r_aero_func(FF, z_FF, z_T, z0, z0t, V_veh, N_v, num_veh, a_traffic)

    # Should still work, traffic contribution will be minimal
    assert r_aero > 0
    assert np.isfinite(r_aero)


def test_single_vehicle_type():
    """Test with single vehicle type."""
    FF = 3.0
    z_FF = 10.0
    z_T = 2.0
    z0 = 0.05
    z0t = 0.005
    V_veh = np.array([70.0])
    N_v = np.array([200.0])
    num_veh = 1
    a_traffic = np.array([0.12])

    r_aero = r_aero_func(FF, z_FF, z_T, z0, z0t, V_veh, N_v, num_veh, a_traffic)

    assert r_aero > 0
    assert np.isfinite(r_aero)


def test_different_roughness_lengths():
    """Test with various roughness length values."""
    FF = 5.0
    z_FF = 10.0
    z_T = 2.0
    V_veh = np.array([80.0])
    N_v = np.array([100.0])
    num_veh = 1
    a_traffic = np.array([0.1])

    # Test different roughness lengths
    roughness_pairs = [
        (0.001, 0.0001),  # Very smooth
        (0.01, 0.001),  # Typical road
        (0.1, 0.01),  # Rough surface
        (1.0, 0.1),  # Very rough
    ]

    results = []
    for z0, z0t in roughness_pairs:
        r_aero = r_aero_func(FF, z_FF, z_T, z0, z0t, V_veh, N_v, num_veh, a_traffic)
        results.append(r_aero)
        assert r_aero > 0
        assert np.isfinite(r_aero)

    # Higher roughness should generally lead to lower resistance
    assert results[0] > results[-1]


def test_numerical_consistency():
    """Test numerical consistency with known calculation."""
    FF = 5.0
    z_FF = 10.0
    z_T = 2.0
    z0 = 0.01
    z0t = 0.001
    V_veh = np.array([80.0, 60.0])
    N_v = np.array([100.0, 50.0])
    num_veh = 2
    a_traffic = np.array([0.1, 0.08])

    r_aero = r_aero_func(FF, z_FF, z_T, z0, z0t, V_veh, N_v, num_veh, a_traffic)

    # Manual calculation
    kappa = 0.4
    inv_r_wind = FF * kappa**2 / (np.log(z_FF / z0) * np.log(z_T / z0t))

    inv_r_traffic = 0.0
    for v in range(num_veh):
        inv_r_traffic += N_v[v] * V_veh[v] * a_traffic[v]
    inv_r_traffic = max(1e-6, inv_r_traffic / 3600 / 3.6)

    expected_r_aero = 1 / (inv_r_traffic + inv_r_wind)

    assert np.isclose(r_aero, expected_r_aero, rtol=1e-10)


def test_edge_case_heights():
    """Test with edge case measurement heights."""
    FF = 5.0
    z0 = 0.01
    z0t = 0.001
    V_veh = np.array([80.0])
    N_v = np.array([100.0])
    num_veh = 1
    a_traffic = np.array([0.1])

    # Test with very low measurement heights
    z_FF = 0.5
    z_T = 0.1
    r_aero_low = r_aero_func(FF, z_FF, z_T, z0, z0t, V_veh, N_v, num_veh, a_traffic)

    # Test with very high measurement heights
    z_FF = 100.0
    z_T = 50.0
    r_aero_high = r_aero_func(FF, z_FF, z_T, z0, z0t, V_veh, N_v, num_veh, a_traffic)

    assert r_aero_low > 0 and np.isfinite(r_aero_low)
    assert r_aero_high > 0 and np.isfinite(r_aero_high)
    # Higher measurement heights should give different resistance
    assert r_aero_low != r_aero_high


def test_array_size_mismatch():
    """Test behavior when array sizes don't match num_veh."""
    FF = 5.0
    z_FF = 10.0
    z_T = 2.0
    z0 = 0.01
    z0t = 0.001
    V_veh = np.array([80.0, 60.0])
    N_v = np.array([100.0, 50.0])
    num_veh = 2  # Matches array size
    a_traffic = np.array([0.1, 0.08])

    # This should work fine
    r_aero = r_aero_func(FF, z_FF, z_T, z0, z0t, V_veh, N_v, num_veh, a_traffic)
    assert r_aero > 0

    # Test with num_veh less than array sizes
    num_veh_small = 1
    r_aero_small = r_aero_func(
        FF, z_FF, z_T, z0, z0t, V_veh, N_v, num_veh_small, a_traffic
    )
    assert r_aero_small > 0  # Should still work, just uses fewer vehicles

    # Test with mismatched array sizes
    N_v_short = np.array([100.0])
    V_veh_short = np.array([80.0])
    a_traffic_short = np.array([0.1])
    r_aero_mismatched = r_aero_func(
        FF, z_FF, z_T, z0, z0t, V_veh_short, N_v_short, num_veh_small, a_traffic_short
    )
    assert r_aero_mismatched > 0  # Should work with matching array sizes


def test_high_traffic_conditions():
    """Test with very high traffic conditions."""
    FF = 5.0
    z_FF = 10.0
    z_T = 2.0
    z0 = 0.01
    z0t = 0.001
    V_veh = np.array([100.0, 80.0, 60.0])
    N_v = np.array([1000.0, 500.0, 200.0])  # High traffic volume
    num_veh = 3
    a_traffic = np.array([0.15, 0.12, 0.08])

    r_aero_high_traffic = r_aero_func(
        FF, z_FF, z_T, z0, z0t, V_veh, N_v, num_veh, a_traffic
    )

    # Compare with low traffic
    N_v_low = np.array([10.0, 5.0, 2.0])
    r_aero_low_traffic = r_aero_func(
        FF, z_FF, z_T, z0, z0t, V_veh, N_v_low, num_veh, a_traffic
    )

    # High traffic should reduce aerodynamic resistance
    assert r_aero_high_traffic < r_aero_low_traffic
