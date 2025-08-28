from input_classes import (
    converted_data,
    input_metadata,
    input_airquality,
    input_meteorology,
    input_activity,
)
from config_classes import (
    model_parameters,
    model_flags,
)
from .shared_plot_data import shared_plot_data
import numpy as np
import constants
from initialise import time_config, model_variables


def init_shared_data(
    *,
    time_config: time_config,
    converted_data: converted_data,
    metadata: input_metadata,
    airquality_data: input_airquality,
    meteo_input: input_meteorology,
    model_parameters: model_parameters,
    model_flags: model_flags,
    model_variables: model_variables,
    input_activity: input_activity,
    ro: int,
    plot_size_fraction: int,
) -> shared_plot_data:
    # Convenience aliases
    mv = model_variables
    mp = model_parameters
    av = (int(model_flags.plot_type_flag),)

    # X-label text based on averaging mode
    if av[0] in (1, 2, 4, 7, 8):
        xlabel_text = "Date"
    elif av[0] == 3:
        xlabel_text = "Hour"
    elif av[0] == 5:
        xlabel_text = "Day"
    else:
        xlabel_text = "Date"

    # Dimensions
    n_date = converted_data.n_date
    n_tracks = mp.num_track

    # Track weights
    f_track = np.asarray(mp.f_track[:n_tracks], dtype=float)
    f_track = f_track / max(f_track.sum(), 1.0)

    # Date vector for selected road
    date_num = converted_data.date_data[constants.datenum_index, :n_date, ro]

    # Width conversion factor (to convert per-km and per-lane quantities)
    b_road_lanes = metadata.b_road_lanes or (metadata.n_lanes * metadata.b_lane)
    b_factor = 1.0 / 1000.0 / max(b_road_lanes, 1e-6)

    # Sum across tracks for emissions, concentrations, and mass data
    C_data_sum_tracks = mv.C_data[:, :, :, :n_date, :n_tracks, ro].sum(axis=4)
    E_road_data_sum_tracks = mv.E_road_data[:, :, :, :n_date, :n_tracks, ro].sum(axis=4)
    M_road_data_sum_tracks = mv.M_road_data[:, :, :n_date, :n_tracks, ro].sum(axis=3)
    M_road_balance_data_sum_tracks = mv.M_road_balance_data[
        :, :, :, :n_date, :n_tracks, ro
    ].sum(axis=4)
    WR_time_sum_tracks = mv.WR_time_data[:, :n_date, :n_tracks, ro].sum(axis=2)

    # Weighted averages across tracks for road/ground state and related factors
    # road_meteo_weighted: [num_road_meteo, n_date]
    road_meteo = mv.road_meteo_data[:, :n_date, :n_tracks, ro]
    road_meteo_weighted = np.tensordot(road_meteo, f_track, axes=([2], [0]))

    # g_road_weighted: [num_moisture, n_date]
    g_road = mv.g_road_data[:, :n_date, :n_tracks, ro]
    g_road_weighted = np.tensordot(g_road, f_track, axes=([2], [0]))

    # g_road_balance_weighted: [num_moisture, num_moistbalance, n_date]
    g_road_bal = mv.g_road_balance_data[:, :, :n_date, :n_tracks, ro]
    g_road_balance_weighted = np.tensordot(g_road_bal, f_track, axes=([3], [0]))

    # road_salt_weighted: [num_saltdata, num_salt, n_date]
    road_salt = mv.road_salt_data[:, :, :n_date, :n_tracks, ro]
    road_salt_weighted = np.tensordot(road_salt, f_track, axes=([3], [0]))

    # f_q_weighted: [num_source_all, n_date]
    f_q = mv.f_q[:, :n_date, :n_tracks, ro]
    f_q_weighted = np.tensordot(f_q, f_track, axes=([2], [0]))

    # f_q_obs_weighted: [n_date]
    f_q_obs = mv.f_q_obs[:n_date, :n_tracks, ro]
    f_q_obs_weighted = np.tensordot(f_q_obs, f_track, axes=([1], [0]))

    # Per-road input data
    meteo_data_ro = converted_data.meteo_data[:, :n_date, ro]
    traffic_data_ro = converted_data.traffic_data[:, :n_date, ro]
    activity_data_ro = converted_data.activity_data[:, :n_date, ro]
    f_conc = mv.f_conc[:n_date, ro]

    # Observations
    PM_obs_net = np.array(airquality_data.PM_obs_net, copy=True)
    PM_obs_bg = np.array(airquality_data.PM_obs_bg, copy=True)
    Salt_obs = np.array(airquality_data.Salt_obs, copy=True)
    Salt_obs_available = np.array(airquality_data.Salt_obs_available, copy=True)
    NOX_obs = np.array(airquality_data.NOX_obs, copy=True)
    NOX_background = np.array(airquality_data.NOX_background, copy=True)
    NOX_obs_net = np.array(airquality_data.NOX_obs_net, copy=True)

    # Derived course fraction handling (virtual size: PM_course = PM10 - PM2.5)
    pm_course_derived = plot_size_fraction == constants.pm_course
    C_data_course = np.zeros((0, 0, 0, 0))
    E_road_data_course = np.zeros((0, 0, 0, 0))
    M_road_data_course = np.zeros((0, 0, 0))
    M_road_balance_data_course = np.zeros((0, 0, 0, 0))
    PM_obs_net_course = np.zeros((0, 0))
    PM_obs_bg_course = np.zeros((0, 0))

    if pm_course_derived:
        # Create derived arrays without changing original shapes
        # Shapes match the original but with size dimension collapsed to a single derived fraction
        # C, E: [num_source_all, 1, num_process, n_date]
        # M: [num_source_all, 1, n_date]
        # M balance: [num_source_all, 1, num_dustbalance, n_date]
        def _pick(arr: np.ndarray, i10: int, i25: int) -> np.ndarray:
            return arr[:, i10, ...] - arr[:, i25, ...]

        i10 = constants.pm_10
        i25 = constants.pm_25

        C_course = _pick(C_data_sum_tracks, i10, i25)
        E_course = _pick(E_road_data_sum_tracks, i10, i25)
        M_course = _pick(M_road_data_sum_tracks, i10, i25)
        MB_course = _pick(M_road_balance_data_sum_tracks, i10, i25)

        # Add a singleton size axis at position 1 to keep a consistent interface
        C_data_course = C_course[:, np.newaxis, :, :]
        E_road_data_course = E_course[:, np.newaxis, :, :]
        M_road_data_course = M_course[:, np.newaxis, :]
        M_road_balance_data_course = MB_course[:, np.newaxis, :, :]

        # Observations derived
        if PM_obs_net.shape[1] >= n_date:
            PM_obs_net_course = (
                PM_obs_net[constants.pm_10, :n_date]
                - PM_obs_net[constants.pm_25, :n_date]
            )
            PM_obs_bg_course = (
                PM_obs_bg[constants.pm_10, :n_date]
                - PM_obs_bg[constants.pm_25, :n_date]
            )

    # Availability flags from parsed meteorology input
    road_wetness_obs_available = meteo_input.road_wetness_obs_available
    road_temperature_obs_available = meteo_input.road_temperature_obs_available
    road_wetness_obs_in_mm = meteo_input.road_wetness_obs_in_mm

    return shared_plot_data(
        av=av,
        i_min=time_config.min_time,
        i_max=time_config.max_time,
        dt=float(time_config.dt),
        xlabel_text=xlabel_text,
        date_num=date_num,
        b_factor=b_factor,
        plot_size_fraction=plot_size_fraction,
        nodata=metadata.nodata,
        C_data_sum_tracks=C_data_sum_tracks,
        E_road_data_sum_tracks=E_road_data_sum_tracks,
        M_road_data_sum_tracks=M_road_data_sum_tracks,
        M_road_balance_data_sum_tracks=M_road_balance_data_sum_tracks,
        WR_time_sum_tracks=WR_time_sum_tracks,
        road_meteo_weighted=road_meteo_weighted,
        g_road_weighted=g_road_weighted,
        g_road_balance_weighted=g_road_balance_weighted,
        road_salt_weighted=road_salt_weighted,
        f_q_weighted=f_q_weighted,
        f_q_obs_weighted=f_q_obs_weighted,
        meteo_data_ro=meteo_data_ro,
        traffic_data_ro=traffic_data_ro,
        activity_data_ro=activity_data_ro,
        f_conc=f_conc,
        PM_obs_net=PM_obs_net,
        PM_obs_bg=PM_obs_bg,
        Salt_obs=Salt_obs,
        Salt_obs_available=Salt_obs_available,
        NOX_obs=NOX_obs,
        NOX_background=NOX_background,
        NOX_obs_net=NOX_obs_net,
        pm_course_derived=pm_course_derived,
        C_data_course=C_data_course,
        E_road_data_course=E_road_data_course,
        M_road_data_course=M_road_data_course,
        M_road_balance_data_course=M_road_balance_data_course,
        PM_obs_net_course=PM_obs_net_course,
        PM_obs_bg_course=PM_obs_bg_course,
        road_wetness_obs_available=road_wetness_obs_available,
        road_wetness_obs_in_mm=road_wetness_obs_in_mm,
        road_temperature_obs_available=road_temperature_obs_available,
        use_energy_correction_flag=model_flags.use_energy_correction_flag,
        use_wetting_data_flag=model_flags.use_wetting_data_flag,
        salt2_str=input_activity.salt2_str,
    )
