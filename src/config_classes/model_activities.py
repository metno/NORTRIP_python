from dataclasses import dataclass, field
from typing import List


@dataclass
class model_activities:
    """
    Dataclass containing model activities.
    """

    # Salting activities
    salting_hour: List[int] = field(default_factory=lambda: [0, 0])
    delay_salting_day: float = 0.9
    check_salting_day: float = 0.5
    min_temp_salt: float = -6.0
    max_temp_salt: float = 0.0
    precip_rule_salt: float = 0.1
    RH_rule_salt: float = 90.0
    g_salting_rule: float = 0.1
    salt_mass: float = 0.1
    salt_dilution: float = 0.2
    salt_type_distribution: int = 1

    # Sanding activities
    sanding_hour: List[int] = field(default_factory=lambda: [0, 0])
    delay_sanding_day: float = 0.9
    check_sanding_day: float = 0.5
    min_temp_sand: float = -6.0
    max_temp_sand: float = 0.0
    precip_rule_sand: float = 0.1
    RH_rule_sand: float = 90.0
    g_sanding_rule: float = 0.1
    sand_mass: float = 0.1
    sand_dilution: float = 0.2

    # Ploughing activities
    delay_ploughing_hour: float = 3.0
    ploughing_thresh: float = 0.0  # Note: default depends on existing value in MATLAB

    # Cleaning activities
    delay_cleaning_hour: float = 168.0  # 7*24 hours
    min_temp_cleaning: float = 0.0
    clean_with_salting: int = 0
    start_month_cleaning: int = 1
    end_month_cleaning: int = 12
    wetting_with_cleaning: int = 0
    efficiency_of_cleaning: float = 0.0

    # Binding activities
    binding_hour: List[int] = field(default_factory=lambda: [0, 0])
    delay_binding_day: float = 0.9
    check_binding_day: float = 0.5
    min_temp_binding: float = -6.0
    max_temp_binding: float = 0.0
    precip_rule_binding: float = 0.1
    RH_rule_binding: float = 90.0
    g_binding_rule: float = 0.1
    binding_mass: float = 0.1
    binding_dilution: float = 0.2
    start_month_binding: int = 1
    end_month_binding: int = 12
