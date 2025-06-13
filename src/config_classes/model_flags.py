from dataclasses import dataclass


@dataclass
class model_flags:
    """
    Dataclass containing model flags and their default values.
    """

    road_wear_flag: int = 1
    tyre_wear_flag: int = 1
    brake_wear_flag: int = 1
    road_suspension_flag: int = 1
    wind_suspension_flag: int = 0
    evaporation_flag: int = 1
    plot_type_flag: int = 1
    save_type_flag: int = 1
    use_multiple_save_dates_flag: int = 0
    abrasion_flag: int = 0
    crushing_flag: int = 0
    exhaust_flag: int = 1
    retention_flag: int = 1
    dust_drainage_flag: int = 2
    dust_ploughing_flag: int = 1
    dust_deposition_flag: int = 0
    dust_spray_flag: int = 0
    use_obs_retention_flag: int = 0
    canyon_shadow_flag: int = 0
    canyon_long_rad_flag: int = 0
    auto_salting_flag: int = 0
    auto_binding_flag: int = 0
    auto_sanding_flag: int = 0
    auto_ploughing_flag: int = 0
    auto_cleaning_flag: int = 0
    use_stability_flag: int = 1
    use_salting_data_1_flag: int = 1
    use_salting_data_2_flag: int = 1
    use_sanding_data_flag: int = 0
    use_ploughing_data_flag: int = 0
    use_cleaning_data_flag: int = 0
    use_wetting_data_flag: int = 0
    water_spray_flag: int = 0
    drainage_type_flag: int = 2
    surface_humidity_flag: int = 1
    use_salt_humidity_flag: int = 0
    use_subsurface_flag: int = 1
    use_energy_correction_flag: int = 1
    forecast_hour: int = 0
    use_traffic_turb_flag: int = 1
    use_ospm_flag: int = 0
