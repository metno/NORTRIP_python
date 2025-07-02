from src.functions.r_aero_func import r_aero_func


def test_r_aero_func():
    """Test aerodynamic resistance calculation."""

    # Test basic case
    FF = 5.0  # Wind speed
    z_FF = 10.0  # Wind measurement height
    z_T = 2.0  # Temperature measurement height
    z0 = 0.1  # Roughness length for momentum
    z0t = 0.01  # Roughness length for heat
    V_veh = [50.0, 30.0]  # Vehicle velocities
    N_v = [100.0, 200.0]  # Number of vehicles
    num_veh = 2
    a_traffic = [0.1, 0.05]  # Traffic coefficients

    result = r_aero_func(FF, z_FF, z_T, z0, z0t, V_veh, N_v, num_veh, a_traffic)

    # Result should be positive resistance
    assert result > 0

    # Test with higher wind speed (should give lower resistance)
    result_high_wind = r_aero_func(
        10.0, z_FF, z_T, z0, z0t, V_veh, N_v, num_veh, a_traffic
    )
    assert result_high_wind < result
    assert result_high_wind > 0


def test_r_aero_func_wind_component():
    """Test wind component of aerodynamic resistance."""

    # Test with no traffic (only wind component)
    FF = 8.0
    z_FF = 10.0
    z_T = 2.0
    z0 = 0.05
    z0t = 0.005
    V_veh = [0.0]  # No vehicles
    N_v = [0.0]
    num_veh = 1
    a_traffic = [0.0]

    result = r_aero_func(FF, z_FF, z_T, z0, z0t, V_veh, N_v, num_veh, a_traffic)

    # Calculate expected wind component manually
    import numpy as np

    kappa = 0.4
    expected_inv_r_wind = (
        max(FF, 0.2) * kappa**2 / (np.log(z_FF / z0) * np.log(z_T / z0t))
    )
    # Account for minimum traffic resistance
    expected_inv_r_traffic = max(1e-6, 0.0)  # Zero traffic but with minimum
    expected_inv_r_aero = expected_inv_r_wind + expected_inv_r_traffic
    expected_r_aero = 1 / expected_inv_r_aero

    assert abs(result - expected_r_aero) < 1e-6


def test_r_aero_func_minimum_wind():
    """Test minimum wind speed handling."""

    # Test with very low wind speed (should use minimum of 0.2)
    FF_low = 0.05  # Very low wind
    z_FF = 10.0
    z_T = 2.0
    z0 = 0.1
    z0t = 0.01
    V_veh = [0.0]
    N_v = [0.0]
    num_veh = 1
    a_traffic = [0.0]

    result_low = r_aero_func(FF_low, z_FF, z_T, z0, z0t, V_veh, N_v, num_veh, a_traffic)

    # Test with minimum wind speed
    result_min = r_aero_func(0.2, z_FF, z_T, z0, z0t, V_veh, N_v, num_veh, a_traffic)

    # Results should be the same (both use 0.2 m/s)
    assert abs(result_low - result_min) < 1e-10


def test_r_aero_func_traffic_component():
    """Test traffic component of aerodynamic resistance."""

    FF = 3.0
    z_FF = 10.0
    z_T = 2.0
    z0 = 0.1
    z0t = 0.01

    # Test with significant traffic
    V_veh = [60.0, 40.0, 80.0]  # km/h
    N_v = [50.0, 100.0, 30.0]  # vehicles/hour
    num_veh = 3
    a_traffic = [0.2, 0.15, 0.25]

    result_with_traffic = r_aero_func(
        FF, z_FF, z_T, z0, z0t, V_veh, N_v, num_veh, a_traffic
    )

    # Test with no traffic
    V_veh_no = [0.0]
    N_v_no = [0.0]
    num_veh_no = 1
    a_traffic_no = [0.0]

    result_no_traffic = r_aero_func(
        FF, z_FF, z_T, z0, z0t, V_veh_no, N_v_no, num_veh_no, a_traffic_no
    )

    # Traffic should reduce resistance (increase inverse resistance)
    assert result_with_traffic < result_no_traffic


def test_r_aero_func_multiple_vehicle_types():
    """Test with multiple vehicle types."""

    FF = 6.0
    z_FF = 10.0
    z_T = 2.0
    z0 = 0.05
    z0t = 0.005

    # Test different vehicle configurations
    V_veh = [50.0, 70.0, 30.0]  # Different speeds
    N_v = [80.0, 40.0, 120.0]  # Different counts
    num_veh = 3
    a_traffic = [0.1, 0.2, 0.05]  # Different coefficients

    result = r_aero_func(FF, z_FF, z_T, z0, z0t, V_veh, N_v, num_veh, a_traffic)

    # Calculate expected traffic contribution manually
    expected_inv_r_traffic = 0.0
    for v in range(num_veh):
        expected_inv_r_traffic += N_v[v] * V_veh[v] * a_traffic[v]
    expected_inv_r_traffic = max(1e-6, expected_inv_r_traffic / 3600 / 3.6)

    # Result should be positive and reasonable
    assert result > 0
    assert result < 1000  # Reasonable upper bound for resistance


def test_r_aero_func_roughness_effects():
    """Test effects of different roughness lengths."""

    FF = 7.0
    z_FF = 10.0
    z_T = 2.0
    V_veh = [40.0]
    N_v = [50.0]
    num_veh = 1
    a_traffic = [0.1]

    # Smooth surface (low roughness)
    z0_smooth = 0.01
    z0t_smooth = 0.001
    result_smooth = r_aero_func(
        FF, z_FF, z_T, z0_smooth, z0t_smooth, V_veh, N_v, num_veh, a_traffic
    )

    # Rough surface (high roughness)
    z0_rough = 0.5
    z0t_rough = 0.05
    result_rough = r_aero_func(
        FF, z_FF, z_T, z0_rough, z0t_rough, V_veh, N_v, num_veh, a_traffic
    )

    # Smooth surface should have higher resistance (lower transfer)
    assert result_smooth > result_rough
    assert result_smooth > 0
    assert result_rough > 0


def test_r_aero_func_height_effects():
    """Test effects of measurement heights."""

    FF = 5.0
    z0 = 0.1
    z0t = 0.01
    V_veh = [50.0]
    N_v = [75.0]
    num_veh = 1
    a_traffic = [0.12]

    # Low measurement heights
    result_low = r_aero_func(FF, 5.0, 1.0, z0, z0t, V_veh, N_v, num_veh, a_traffic)

    # High measurement heights
    result_high = r_aero_func(FF, 20.0, 5.0, z0, z0t, V_veh, N_v, num_veh, a_traffic)

    # Higher measurement heights should give higher resistance
    assert result_high > result_low
    assert result_low > 0
    assert result_high > 0


def test_r_aero_func_minimum_traffic():
    """Test minimum traffic resistance handling."""

    FF = 4.0
    z_FF = 10.0
    z_T = 2.0
    z0 = 0.1
    z0t = 0.01

    # Very small traffic contribution
    V_veh = [1.0]  # Very low speed
    N_v = [0.1]  # Very few vehicles
    num_veh = 1
    a_traffic = [0.001]  # Very small coefficient

    result = r_aero_func(FF, z_FF, z_T, z0, z0t, V_veh, N_v, num_veh, a_traffic)

    # Should still give reasonable result (minimum traffic resistance applied)
    assert result > 0
    assert result < 1000
