from config_classes.model_parameters import model_parameters
import pandas as pd
import logging
import constants
from pd_util import safe_float

logger = logging.getLogger(__name__)


def read_model_parameters(parameter_df: pd.DataFrame) -> model_parameters:
    """
    Load model parameters from xlsx file and return an instance of model_parameters.

    Args:
        paramter_df (DataFrame): DataFrame containing model parameters.

    Returns:
        loaded_parameters (model_paramters): An instance of model_parameters with loaded values.
    """
    loaded_parameters = model_parameters()

    # Extract header and data series for searching
    header_series = parameter_df.iloc[:, 0]  # First column contains headers
    text = ""
    # Road wear parameters
    for s in range(constants.num_wear):
        if s == constants.road_index:
            text = "Road wear"
        elif s == constants.tyre_index:
            text = "Tyre wear"
        elif s == constants.brake_index:
            text = "Brake wear"

        # Find the row with the wear type text
        k2 = header_series.str.contains(text, case=False, na=False)
        if k2.any():
            row_idx = int(k2.idxmax())
            # W_0 values for different tyre and vehicle types
            loaded_parameters.W_0[s, constants.st, constants.he] = safe_float(
                parameter_df.iloc[row_idx + 2, 1]
            )
            loaded_parameters.W_0[s, constants.wi, constants.he] = safe_float(
                parameter_df.iloc[row_idx + 2, 2]
            )
            loaded_parameters.W_0[s, constants.su, constants.he] = safe_float(
                parameter_df.iloc[row_idx + 2, 3]
            )
            loaded_parameters.W_0[s, constants.st, constants.li] = safe_float(
                parameter_df.iloc[row_idx + 3, 1]
            )
            loaded_parameters.W_0[s, constants.wi, constants.li] = safe_float(
                parameter_df.iloc[row_idx + 3, 2]
            )
            loaded_parameters.W_0[s, constants.su, constants.li] = safe_float(
                parameter_df.iloc[row_idx + 3, 3]
            )

            # a_wear coefficients
            for i in range(5):
                loaded_parameters.a_wear[s, i] = safe_float(
                    parameter_df.iloc[row_idx + 5, 1 + i]
                )

    # Snow depth wear threshold
    snow_idx = header_series.str.contains(
        "Snow depth wear threshold", case=False, na=False
    )
    if snow_idx.any():
        row_idx = int(snow_idx.idxmax())
        loaded_parameters.s_roadwear_thresh = safe_float(
            parameter_df.iloc[row_idx + 2, 1]
        )

    # Pavement type scaling factors
    pave_idx = header_series.str.contains(
        "Pavement type scaling factor", case=False, na=False
    )
    if pave_idx.any():
        row_idx = int(pave_idx.idxmax())
        loaded_parameters.num_pave = int(safe_float(parameter_df.iloc[row_idx + 1, 1]))
        loaded_parameters.h_pave_str = []
        loaded_parameters.h_pave = []
        for p in range(loaded_parameters.num_pave):
            loaded_parameters.h_pave_str.append(
                str(parameter_df.iloc[row_idx + 3 + p, 1])
            )
            loaded_parameters.h_pave.append(
                safe_float(parameter_df.iloc[row_idx + 3 + p, 2])
            )

    # Driving cycle scaling factors
    dc_idx = header_series.str.contains(
        "Driving cycle scaling factor", case=False, na=False
    )
    if dc_idx.any():
        row_idx = int(dc_idx.idxmax())
        loaded_parameters.num_dc = int(safe_float(parameter_df.iloc[row_idx + 1, 1]))
        loaded_parameters.h_drivingcycle_str = []
        loaded_parameters.h_drivingcycle = []
        for d in range(loaded_parameters.num_dc):
            loaded_parameters.h_drivingcycle_str.append(
                str(parameter_df.iloc[row_idx + 3 + d, 1])
            )
            loaded_parameters.h_drivingcycle.append(
                safe_float(parameter_df.iloc[row_idx + 3 + d, 2])
            )

    # Suspension scaling factors
    sus_idx = header_series.str.contains(
        "Suspension scaling factors", case=False, na=False
    )
    if sus_idx.any():
        row_idx = int(sus_idx.idxmax())
        for x in range(constants.num_size):
            for s in range(constants.num_source):
                loaded_parameters.h_0_sus[s, x] = safe_float(
                    parameter_df.iloc[row_idx + 2 + s, 1 + x]
                )
            loaded_parameters.h_0_q_road[x] = safe_float(
                parameter_df.iloc[row_idx + 2 + constants.num_source, 1 + x]
            )

    # Road suspension factors
    road_sus_idx = header_series.str.contains("Road suspension", case=False, na=False)
    if road_sus_idx.any():
        row_idx = int(road_sus_idx.idxmax())
        loaded_parameters.f_0_suspension[0, 0, constants.st, constants.he] = safe_float(
            parameter_df.iloc[row_idx + 2, 1]
        )
        loaded_parameters.f_0_suspension[0, 0, constants.wi, constants.he] = safe_float(
            parameter_df.iloc[row_idx + 2, 2]
        )
        loaded_parameters.f_0_suspension[0, 0, constants.su, constants.he] = safe_float(
            parameter_df.iloc[row_idx + 2, 3]
        )
        loaded_parameters.f_0_suspension[0, 0, constants.st, constants.li] = safe_float(
            parameter_df.iloc[row_idx + 3, 1]
        )
        loaded_parameters.f_0_suspension[0, 0, constants.wi, constants.li] = safe_float(
            parameter_df.iloc[row_idx + 3, 2]
        )
        loaded_parameters.f_0_suspension[0, 0, constants.su, constants.li] = safe_float(
            parameter_df.iloc[row_idx + 3, 3]
        )

        for i in range(5):
            loaded_parameters.a_sus[i] = safe_float(
                parameter_df.iloc[row_idx + 5, 1 + i]
            )

        # Fill suspension matrix
        for s in range(constants.num_source):
            for x in range(constants.num_size):
                for t in range(constants.num_tyre):
                    for v in range(constants.num_veh):
                        loaded_parameters.f_0_suspension[s, x, t, v] = (
                            loaded_parameters.f_0_suspension[0, 0, t, v]
                            * loaded_parameters.h_0_sus[s, x]
                        )

    # Abrasion factors
    abr_idx = header_series.str.contains("Abrasion factor", case=False, na=False)
    if abr_idx.any():
        row_idx = int(abr_idx.idxmax())
        loaded_parameters.f_0_abrasion[constants.st, constants.he] = safe_float(
            parameter_df.iloc[row_idx + 2, 1]
        )
        loaded_parameters.f_0_abrasion[constants.wi, constants.he] = safe_float(
            parameter_df.iloc[row_idx + 2, 2]
        )
        loaded_parameters.f_0_abrasion[constants.su, constants.he] = safe_float(
            parameter_df.iloc[row_idx + 2, 3]
        )
        loaded_parameters.f_0_abrasion[constants.st, constants.li] = safe_float(
            parameter_df.iloc[row_idx + 3, 1]
        )
        loaded_parameters.f_0_abrasion[constants.wi, constants.li] = safe_float(
            parameter_df.iloc[row_idx + 3, 2]
        )
        loaded_parameters.f_0_abrasion[constants.su, constants.li] = safe_float(
            parameter_df.iloc[row_idx + 3, 3]
        )
        loaded_parameters.V_ref_abrasion = safe_float(parameter_df.iloc[row_idx + 4, 1])

        for x in range(constants.num_size):
            loaded_parameters.h_0_abrasion[x] = safe_float(
                parameter_df.iloc[row_idx + 6, 1 + x]
            )

    # Crushing factors
    crush_idx = header_series.str.contains("Crushing factor", case=False, na=False)
    if crush_idx.any():
        row_idx = int(crush_idx.idxmax())
        loaded_parameters.f_0_crushing[constants.st, constants.he] = safe_float(
            parameter_df.iloc[row_idx + 2, 1]
        )
        loaded_parameters.f_0_crushing[constants.wi, constants.he] = safe_float(
            parameter_df.iloc[row_idx + 2, 2]
        )
        loaded_parameters.f_0_crushing[constants.su, constants.he] = safe_float(
            parameter_df.iloc[row_idx + 2, 3]
        )
        loaded_parameters.f_0_crushing[constants.st, constants.li] = safe_float(
            parameter_df.iloc[row_idx + 3, 1]
        )
        loaded_parameters.f_0_crushing[constants.wi, constants.li] = safe_float(
            parameter_df.iloc[row_idx + 3, 2]
        )
        loaded_parameters.f_0_crushing[constants.su, constants.li] = safe_float(
            parameter_df.iloc[row_idx + 3, 3]
        )
        loaded_parameters.V_ref_crushing = safe_float(parameter_df.iloc[row_idx + 4, 1])

        for x in range(constants.num_size):
            loaded_parameters.h_0_crushing[x] = safe_float(
                parameter_df.iloc[row_idx + 6, 1 + x]
            )

    # Source participation in abrasion and crushing
    part_idx = header_series.str.contains(
        "Sources participating in abrasion and crushing", case=False, na=False
    )
    if part_idx.any():
        row_idx = int(part_idx.idxmax())
        for s in range(constants.num_source):
            loaded_parameters.p_0_abrasion[s] = safe_float(
                parameter_df.iloc[row_idx + 2 + s, 1]
            )
            loaded_parameters.p_0_crushing[s] = safe_float(
                parameter_df.iloc[row_idx + 2 + s, 2]
            )

    # Direct emission factors
    dir_idx = header_series.str.contains("Direct emission factor", case=False, na=False)
    if dir_idx.any():
        row_idx = int(dir_idx.idxmax())
        for s in range(constants.num_wear):
            loaded_parameters.f_0_dir[s] = safe_float(
                parameter_df.iloc[row_idx + 2 + s, 1]
            )
        loaded_parameters.f_0_dir[constants.crushing_index] = safe_float(
            parameter_df.iloc[row_idx + 2 + constants.num_wear, 1]
        )
        loaded_parameters.f_0_dir[constants.abrasion_index] = safe_float(
            parameter_df.iloc[row_idx + 3 + constants.num_wear, 1]
        )
        loaded_parameters.f_0_dir[constants.exhaust_index] = safe_float(
            parameter_df.iloc[row_idx + 4 + constants.num_wear, 1]
        )

    # Fractional size distribution
    frac_idx = header_series.str.contains(
        "Fractional size distribution", case=False, na=False
    )
    if frac_idx.any():
        row_idx = int(frac_idx.idxmax())
        for x in range(constants.num_size):
            for s in range(constants.num_source):
                loaded_parameters.f_PM[s, x, 0] = safe_float(
                    parameter_df.iloc[row_idx + 2 + s, 1 + x]
                )
                # Distribute to all tyre types
                for t in range(constants.num_tyre):
                    loaded_parameters.f_PM[s, x, t] = loaded_parameters.f_PM[s, x, 0]

            loaded_parameters.f_PM[constants.crushing_index, x, 0] = safe_float(
                parameter_df.iloc[row_idx + 2 + constants.num_source, 1 + x]
            )
            loaded_parameters.f_PM[constants.abrasion_index, x, 0] = safe_float(
                parameter_df.iloc[row_idx + 3 + constants.num_source, 1 + x]
            )

            for t in range(constants.num_tyre):
                loaded_parameters.f_PM[constants.crushing_index, x, t] = (
                    loaded_parameters.f_PM[constants.crushing_index, x, 0]
                )
                loaded_parameters.f_PM[constants.abrasion_index, x, t] = (
                    loaded_parameters.f_PM[constants.abrasion_index, x, 0]
                )

        # Create differential size array
        loaded_parameters.f_PM_bin = loaded_parameters.f_PM.copy()
        for x in range(constants.num_size - 1):
            loaded_parameters.f_PM_bin[: constants.num_source, x, :] = (
                loaded_parameters.f_PM[: constants.num_source, x, :]
                - loaded_parameters.f_PM[: constants.num_source, x + 1, :]
            )
            loaded_parameters.f_PM_bin[constants.crushing_index, x, :] = (
                loaded_parameters.f_PM[constants.crushing_index, x, :]
                - loaded_parameters.f_PM[constants.crushing_index, x + 1, :]
            )
            loaded_parameters.f_PM_bin[constants.abrasion_index, x, :] = (
                loaded_parameters.f_PM[constants.abrasion_index, x, :]
                - loaded_parameters.f_PM[constants.abrasion_index, x + 1, :]
            )

        loaded_parameters.V_ref_pm_fraction = safe_float(
            parameter_df.iloc[row_idx + 4 + constants.num_source, 1]
        )
        loaded_parameters.c_pm_fraction = safe_float(
            parameter_df.iloc[row_idx + 5 + constants.num_source, 1]
        )

    # Wind blown dust parameters
    wind_idx = header_series.str.contains(
        "Wind blown dust emission factors", case=False, na=False
    )
    if wind_idx.any():
        row_idx = int(wind_idx.idxmax())
        loaded_parameters.tau_wind = safe_float(parameter_df.iloc[row_idx + 2, 1])
        loaded_parameters.FF_thresh = safe_float(parameter_df.iloc[row_idx + 3, 1])

    # Activity efficiency factors
    eff_idx = header_series.str.contains(
        "Activity efficiency factors", case=False, na=False
    )
    if eff_idx.any():
        row_idx = int(eff_idx.idxmax())
        for x in range(constants.num_size):
            loaded_parameters.h_eff[
                constants.ploughing_eff_index, constants.dust_index, x
            ] = safe_float(parameter_df.iloc[row_idx + 2, 1 + x])
            loaded_parameters.h_eff[
                constants.cleaning_eff_index, constants.dust_index, x
            ] = safe_float(parameter_df.iloc[row_idx + 3, 1 + x])
            loaded_parameters.h_eff[
                constants.drainage_eff_index, constants.dust_index, x
            ] = safe_float(parameter_df.iloc[row_idx + 4, 1 + x])
            loaded_parameters.h_eff[
                constants.spraying_eff_index, constants.dust_index, x
            ] = safe_float(parameter_df.iloc[row_idx + 5, 1 + x])

        for i in range(constants.num_salt):
            for eff_type in range(4):
                loaded_parameters.h_eff[eff_type, constants.salt_index[i], :] = (
                    safe_float(parameter_df.iloc[row_idx + 7 + eff_type, 1 + i])
                )

    # Deposition velocities
    dep_idx = header_series.str.contains("Deposition velocity", case=False, na=False)
    if dep_idx.any():
        row_idx = int(dep_idx.idxmax())
        for x in range(constants.num_size):
            loaded_parameters.w_dep[x] = safe_float(
                parameter_df.iloc[row_idx + 2, 1 + x]
            )

    # Concentration conversion limit values
    conc_idx = header_series.str.contains(
        "Concentration conversion limit values", case=False, na=False
    )
    if conc_idx.any():
        row_idx = int(conc_idx.idxmax())
        loaded_parameters.conc_min = safe_float(parameter_df.iloc[row_idx + 2, 1])
        loaded_parameters.emis_min = safe_float(parameter_df.iloc[row_idx + 3, 1])

    # Spray and splash factors
    spray_idx = header_series.str.contains(
        "Spray and splash factors", case=False, na=False
    )
    if spray_idx.any():
        row_idx = int(spray_idx.idxmax())
        for m in range(constants.num_moisture):
            loaded_parameters.R_0_spray[constants.he, m] = safe_float(
                parameter_df.iloc[row_idx + 2, 1 + m]
            )
            loaded_parameters.R_0_spray[constants.li, m] = safe_float(
                parameter_df.iloc[row_idx + 3, 1 + m]
            )
            loaded_parameters.V_ref_spray[m] = safe_float(
                parameter_df.iloc[row_idx + 4, 1 + m]
            )
            loaded_parameters.g_road_sprayable_min[m] = safe_float(
                parameter_df.iloc[row_idx + 5, 1 + m]
            )
            loaded_parameters.a_spray[m] = safe_float(
                parameter_df.iloc[row_idx + 6, 1 + m]
            )
            loaded_parameters.V_thresh_spray[m] = safe_float(
                parameter_df.iloc[row_idx + 7, 1 + m]
            )

    # Drainage parameters
    drain_idx = header_series.str.contains("Drainage parameters", case=False, na=False)
    if drain_idx.any():
        row_idx = int(drain_idx.idxmax())
        loaded_parameters.g_road_drainable_min = safe_float(
            parameter_df.iloc[row_idx + 2, 1]
        )
        loaded_parameters.g_road_drainable_thresh = safe_float(
            parameter_df.iloc[row_idx + 2, 2]
        )
        loaded_parameters.snow_dust_drainage_retainment_limit = safe_float(
            parameter_df.iloc[row_idx + 3, 1]
        )
        loaded_parameters.tau_road_drainage = safe_float(
            parameter_df.iloc[row_idx + 4, 1]
        )

    # Ploughing parameters
    plough_idx = header_series.str.contains(
        "Ploughing parameters", case=False, na=False
    )
    if plough_idx.any():
        row_idx = int(plough_idx.idxmax())
        for m in range(constants.num_moisture):
            loaded_parameters.h_ploughing_moisture[m] = safe_float(
                parameter_df.iloc[row_idx + 2, 1 + m]
            )
            loaded_parameters.ploughing_thresh_moisture[m] = safe_float(
                parameter_df.iloc[row_idx + 3, 1 + m]
            )

    # Energy balance parameters
    energy_idx = header_series.str.contains(
        "Energy balance parameters", case=False, na=False
    )
    if energy_idx.any():
        row_idx = int(energy_idx.idxmax())
        loaded_parameters.g_road_evaporation_thresh = safe_float(
            parameter_df.iloc[row_idx + 2, 1]
        )
        loaded_parameters.z0 = (
            safe_float(parameter_df.iloc[row_idx + 3, 1]) / 1000.0
        )  # Convert mm to m
        loaded_parameters.albedo_snow = safe_float(parameter_df.iloc[row_idx + 4, 1])
        loaded_parameters.dzs = safe_float(parameter_df.iloc[row_idx + 5, 1])
        loaded_parameters.sub_surf_average_time = safe_float(
            parameter_df.iloc[row_idx + 6, 1]
        )

        for i in range(3):
            loaded_parameters.sub_surf_param[i] = safe_float(
                parameter_df.iloc[row_idx + 8, 1 + i]
            )

        loaded_parameters.a_traffic[constants.he] = safe_float(
            parameter_df.iloc[row_idx + 10, 1]
        )
        loaded_parameters.a_traffic[constants.li] = safe_float(
            parameter_df.iloc[row_idx + 10, 2]
        )
        loaded_parameters.H_veh[constants.he] = safe_float(
            parameter_df.iloc[row_idx + 11, 1]
        )
        loaded_parameters.H_veh[constants.li] = safe_float(
            parameter_df.iloc[row_idx + 11, 2]
        )

    # Retention parameters
    ret_idx = header_series.str.contains("Retention parameters", case=False, na=False)
    if ret_idx.any():
        row_idx = int(ret_idx.idxmax())
        loaded_parameters.g_retention_thresh[constants.road_index] = safe_float(
            parameter_df.iloc[row_idx + 2, 1]
        )
        loaded_parameters.g_retention_thresh[constants.brake_index] = safe_float(
            parameter_df.iloc[row_idx + 2, 2]
        )
        loaded_parameters.g_retention_thresh[constants.salt_index[1]] = safe_float(
            parameter_df.iloc[row_idx + 2, 3]
        )
        loaded_parameters.g_retention_min[constants.road_index] = safe_float(
            parameter_df.iloc[row_idx + 3, 1]
        )
        loaded_parameters.g_retention_min[constants.brake_index] = safe_float(
            parameter_df.iloc[row_idx + 3, 2]
        )
        loaded_parameters.g_retention_min[constants.salt_index[1]] = safe_float(
            parameter_df.iloc[row_idx + 3, 3]
        )

    # OSPM parameters (optional)
    ospm_idx = header_series.str.contains("OSPM parameters", case=False, na=False)
    if ospm_idx.any():
        row_idx = int(ospm_idx.idxmax())
        loaded_parameters.f_roof_ospm_override = safe_float(
            parameter_df.iloc[row_idx + 1, 1]
        )
        loaded_parameters.f_turb_ospm_override = safe_float(
            parameter_df.iloc[row_idx + 1, 2]
        )

    # Surface texture parameters
    texture_idx = header_series.str.contains(
        "Surface texture parameters", case=False, na=False
    )
    if texture_idx.any():
        row_idx = int(texture_idx.idxmax())
        for i in range(5):
            loaded_parameters.texture_scaling[i] = safe_float(
                parameter_df.iloc[row_idx + 2 + i, 1]
            )

        # Apply scaling
        loaded_parameters.g_road_drainable_min *= loaded_parameters.texture_scaling[0]
        loaded_parameters.f_0_suspension *= loaded_parameters.texture_scaling[1]
        loaded_parameters.R_0_spray *= loaded_parameters.texture_scaling[2]
        loaded_parameters.h_eff[
            constants.drainage_eff_index, constants.dust_index, :
        ] *= loaded_parameters.texture_scaling[3]
        loaded_parameters.h_eff[
            constants.spraying_eff_index, constants.dust_index, :
        ] *= loaded_parameters.texture_scaling[4]

    # Track parameters
    track_idx = header_series.str.contains(
        "Road track parameters", case=False, na=False
    )
    if track_idx.any():
        row_idx = int(track_idx.idxmax())
        loaded_parameters.num_track = 0
        temp_f_track = []
        temp_veh_track = []
        temp_mig_track = []
        temp_track_type = []

        for i in [
            constants.alltrack_type,
            constants.outtrack_type,
            constants.intrack_type,
            constants.shoulder_type,
            constants.kerb_type,
        ]:
            include_track = safe_float(parameter_df.iloc[row_idx + 2 + i, 1])
            if include_track:
                loaded_parameters.num_track += 1
                temp_f_track.append(safe_float(parameter_df.iloc[row_idx + 2 + i, 2]))
                temp_veh_track.append(safe_float(parameter_df.iloc[row_idx + 2 + i, 3]))
                temp_mig_track.append(safe_float(parameter_df.iloc[row_idx + 2 + i, 4]))
                temp_track_type.append(i)

        loaded_parameters.f_track = temp_f_track
        loaded_parameters.veh_track = temp_veh_track
        loaded_parameters.mig_track = temp_mig_track
        loaded_parameters.track_type = temp_track_type

        # Normalize to sum to 1
        if abs(sum(loaded_parameters.f_track) - 1.0) > 1e-6:
            total = sum(loaded_parameters.f_track)
            loaded_parameters.f_track = [f / total for f in loaded_parameters.f_track]

        if abs(sum(loaded_parameters.veh_track) - 1.0) > 1e-6:
            total = sum(loaded_parameters.veh_track)
            loaded_parameters.veh_track = [
                v / total for v in loaded_parameters.veh_track
            ]
    logger.info("Successfully loaded model parameters")
    return loaded_parameters
