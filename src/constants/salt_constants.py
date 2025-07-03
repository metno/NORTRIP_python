"""
Salt constants for NORTRIP model.

Constants set in regard to salt calculations including NaCl, MgCl2, CMA, CaCl2, and PFO.
"""

import numpy as np

# Salt type indices
NA = 0  # NaCl
MG = 1  # MgCl2
CMA = 2  # CMA (Calcium Magnesium Acetate)
CA = 3  # CaCl2
PFO = 4  # PFO (Potassium Formate)

# Number of salt types
NUM_SALT = 5

# Molecular weights (g/mol)
M_ATOMIC_WATER = 18.015
M_ATOMIC = np.array([58.4, 95.2, 124.0, 111.0, 84.0])  # [Na, Mg, CMA, Ca, PFO]

# Saturated molar solution ratio
SATURATED = np.array([0.086, 0.050, 0.066, 0.065, 0.165])  # [Na, Mg, CMA, Ca, PFO]

# Expected saturated equilibrium relative humidity (calculated, not used directly)
RH_SATURATED = np.array([75.0, 33.0, 40.0, 31.0, 12.0])  # [Na, Mg, CMA, Ca, PFO]

# Antoine constants for vapor pressure calculations
# Ice constants
A_ANTOINE_ICE = 10.3
B_ANTOINE_ICE = 2600.0
C_ANTOINE_ICE = 270.0

# Salt-specific Antoine constants
A_ANTOINE: np.ndarray = np.array(
    [7.40, 7.20, 7.28, 5.8, 10.3 * 0.975], dtype=np.float64
)  # [Na, Mg, CMA, Ca, PFO]
B_ANTOINE = np.array([1566.0, 1581.0, 1581.0, 1087.0, 2600.0])  # [Na, Mg, CMA, Ca, PFO]
C_ANTOINE = np.array([228.0, 225.0, 225.0, 198.0, 270.0])  # [Na, Mg, CMA, Ca, PFO]

# Saturated melt/freezing temperatures (°C)
MELT_TEMPERATURE_SATURATED = np.array(
    [-21.0, -33.0, -27.5, -51.0, -51.0]
)  # [Na, Mg, CMA, Ca, PFO]

# Over-saturated melt temperatures (°C)
MELT_TEMPERATURE_OVERSATURATED = np.array(
    [-1.0, -15.0, -12.0, -1.0, -25.0]
)  # [Na, Mg, CMA, Ca, PFO]

# Salt saturation factors
F_SALT_SAT = np.array([1.17, 1.5, 1.5, 1.4, 1.5])  # [Na, Mg, CMA, Ca, PFO]

# Over-saturated concentrations
OVER_SATURATED = F_SALT_SAT * SATURATED

# Salt power values for curves
SALT_POWER_VAL = np.array([1.3, 1.3, 1.2, 1.6, 1.2])  # [Na, Mg, CMA, Ca, PFO]

# Vapor pressure corrections
VP_CORRECTION = np.array([0.035, 0.11, 0.17, 0.001, 0.012])  # [Na, Mg, CMA, Ca, PFO]

# Saturated mass ratios (for testing, not used in main program)
SATURATED_MASS = np.array([0.233, 0.216, 0.325, 0.298, 0.48])  # [Na, Mg, CMA, Ca, PFO]

# Van't Hoff factors
VAN_HOFF = np.array([2, 3, 3, 3, 3])  # [Na, Mg, CMA, Ca, PFO]

# Fractional distribution of oversaturated solution
RH_OVER_SATURATED_FRACTION = np.array(
    [0.25, 0.99, 0.99, 0.99, 0.99]
)  # [Na, Mg, CMA, Ca, PFO]
