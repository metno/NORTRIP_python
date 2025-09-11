
from typing import List, Tuple

import numpy as np
import pandas as pd

import constants
from functions.average_data_func import average_data_func
from initialise import time_config as time_config_cls
from initialise.road_dust_initialise_variables import model_variables as model_vars_cls
from input_classes import (
    converted_data as converted_data_cls,
    input_metadata as input_metadata_cls,
    input_airquality as input_airquality_cls,
    input_activity as input_activity_cls,
)
from config_classes import (
    model_parameters as model_parameters_cls,
    model_flags as model_flags_cls,
)


def _avg(
    date_num: np.ndarray,
    series: np.ndarray,
    i_min: int,
    i_max: int,
    av: List[int],
) -> Tuple[List[str], np.ndarray]:
    av_str, _, av_val = average_data_func(date_num, series, i_min, i_max, av)
    flat = np.asarray(av_val).reshape(-1)
    return av_str, flat


def build_results_dataframe(
    *,
    time_config: time_config_cls,
    converted_data: converted_data_cls,
    metadata: input_metadata_cls,
    airquality_data: input_airquality_cls,
    model_parameters: model_parameters_cls,
    model_flags: model_flags_cls,
    model_variables: model_vars_cls,
    input_activity: input_activity_cls,
    av: List[int] | None = None,
) -> pd.DataFrame:
    """
    Build a pandas DataFrame mirroring MATLAB's aggregated output columns.

    This function computes track-weighted and summed series, applies the selected
    averaging routine, and assembles all columns defined by the MATLAB routine.
    """
    if av is None:
        av = [model_flags.plot_type_flag]

    ro = 0
    i_min = time_config.min_time
    i_max = time_config.max_time
    n_date = converted_data.n_date
    nodata = metadata.nodata

    # Prepare weights over tracks
    num_track = model_parameters.num_track
    weights = np.asarray(model_parameters.f_track, dtype=float)
    if weights.size != num_track:
        weights = np.ones(num_track, dtype=float) / max(1, num_track)
    else:
        wsum = np.sum(weights)
        weights = weights / wsum if wsum != 0 else np.ones_like(weights) / max(1, num_track)

    mv = model_variables
    cd = converted_data

    # Temporary arrays (track-weighted)
    road_meteo_data_temp = np.zeros((constants.num_road_meteo, n_date))
    g_road_data_temp = np.zeros((constants.num_moisture, n_date))
    g_road_balance_data_temp = np.zeros((constants.num_moisture, constants.num_moistbalance, n_date))
    road_salt_data_temp = np.zeros((constants.num_saltdata, constants.num_salt, n_date))
    f_q_temp = np.zeros((constants.num_source_all, n_date))
    f_q_obs_temp = np.zeros(n_date)

    for tr in range(num_track):
        w = weights[tr]
        road_meteo_data_temp += mv.road_meteo_data[:, :, tr, ro] * w
        g_road_data_temp += mv.g_road_data[:, :, tr, ro] * w
        g_road_balance_data_temp += mv.g_road_balance_data[:, :, :, tr, ro] * w
        road_salt_data_temp += mv.road_salt_data[:, :, :, tr, ro] * w
        f_q_temp += mv.f_q[:, :, tr, ro] * w
        f_q_obs_temp += mv.f_q_obs[:, tr, ro] * w

    # Summed across tracks
    C_data_temp = np.sum(mv.C_data[:, :, :, :, :num_track, ro], axis=4)
    E_road_data_temp = np.sum(mv.E_road_data[:, :, :, :, :num_track, ro], axis=4)
    M_road_data_temp = np.sum(mv.M_road_data[:, :, :, :num_track, ro], axis=3)
    M_road_balance_data_temp = np.sum(mv.M_road_balance_data[:, :, :, :, :num_track, ro], axis=4)
    WR_time_data_temp = np.sum(mv.WR_time_data[:, :, :num_track, ro], axis=2)

    # Direct copies from converted data
    meteo_data_temp = cd.meteo_data[:, :, ro]
    traffic_data_temp = cd.traffic_data[:, :, ro]
    activity_data_temp = cd.activity_data[:, :, ro]
    date_num = cd.date_data[constants.datenum_index, :, ro]

    # Conversion factor (g/km -> g/m^2)
    b_road_lanes = metadata.b_road_lanes if metadata.b_road_lanes else metadata.n_lanes * metadata.b_lane
    factor = 1.0 / 1000.0 / max(b_road_lanes, 1e-9)

    # Indices and labels
    x = constants.pm_10
    x_load = constants.pm_200
    x2 = constants.pm_25

    # Helper to compute average series and register column
    cols: dict[str, np.ndarray] = {}
    date_str, _ = _avg(date_num, date_num, i_min, i_max, av)
    cols["Date"] = np.array(date_str)

    # Date/time derived columns
    year_series = cd.date_data[constants.year_index, :, ro]
    month_series = cd.date_data[constants.month_index, :, ro]
    day_series = cd.date_data[constants.day_index, :, ro]
    hour_series = cd.date_data[constants.hour_index, :, ro]
    minute_series = cd.date_data[constants.minute_index, :, ro]

    # Weekday mapping based on datenum
    def matlab_weekday(dnums: np.ndarray) -> np.ndarray:
        from datetime import datetime, timedelta

        def dn_to_dt(dn: float) -> datetime:
            return datetime(1, 1, 1) + timedelta(days=dn - 1)

        wd = []
        for dn in dnums:
            dt = dn_to_dt(float(dn))
            py = dt.weekday()  # Monday=0..Sunday=6
            matlab = ((py + 1) % 7) + 1
            wd.append(matlab)
        return np.asarray(wd, dtype=float)

    weekday_series = matlab_weekday(cd.date_data[constants.datenum_index, :, ro])

    for label, series in (
        ("Year", year_series),
        ("Month", month_series),
        ("Day", day_series),
        ("Hour", hour_series),
        ("Weekday", weekday_series),
        ("Minute", minute_series),
    ):
        _, v = _avg(date_num, np.asarray(series, dtype=float), i_min, i_max, av)
        cols[label] = v

    # Traffic
    for label, idx in (
        ("N(total)", constants.N_total_index),
        ("N(he)", constants.N_v_index[constants.he]),
        ("N(li)", constants.N_v_index[constants.li]),
        ("N(st,he)", constants.N_t_v_index[(constants.st, constants.he)]),
        ("N(st,li)", constants.N_t_v_index[(constants.st, constants.li)]),
        ("N(wi,he)", constants.N_t_v_index[(constants.wi, constants.he)]),
        ("N(wi,li)", constants.N_t_v_index[(constants.wi, constants.li)]),
        ("N(su,he)", constants.N_t_v_index[(constants.su, constants.he)]),
        ("N(su,li)", constants.N_t_v_index[(constants.su, constants.li)]),
        ("V_veh(he) (km/hr)", constants.V_veh_index[constants.he]),
        ("V_veh(li) (km/hr)", constants.V_veh_index[constants.li]),
    ):
        _, v = _avg(date_num, traffic_data_temp[idx, :], i_min, i_max, av)
        cols[label] = v

    # Meteorology and energy balance
    meteo_cols = [
        ("T2m (C)", meteo_data_temp[constants.T_a_index, :]),
        ("Ts_road (C)", road_meteo_data_temp[constants.T_s_index, :]),
        ("Ts_road_obs (C)", road_meteo_data_temp[constants.road_temperature_obs_index, :]),
        ("T_sub_mod (C)", road_meteo_data_temp[constants.T_sub_index, :]),
        ("RH (%)", meteo_data_temp[constants.RH_index, :]),
        ("RHs_road (%)", road_meteo_data_temp[constants.RH_s_index, :]),
        ("FF (m/s)", meteo_data_temp[constants.FF_index, :]),
        ("Rain (mm/hr)", meteo_data_temp[constants.Rain_precip_index, :]),
        ("Snow (mm/hr)", meteo_data_temp[constants.Snow_precip_index, :]),
        ("Cloud cover", meteo_data_temp[constants.cloud_cover_index, :]),
        ("Net short rad (W/m^2)", road_meteo_data_temp[constants.short_rad_net_index, :]),
        ("Net long rad (W/m^2)", road_meteo_data_temp[constants.long_rad_net_index, :]),
        ("Incoming long rad (W/m^2)", meteo_data_temp[constants.long_rad_in_index, :]),
        ("Incoming short rad (W/m^2)", meteo_data_temp[constants.short_rad_in_index, :]),
        ("Sensible_road (W/m^2)", road_meteo_data_temp[constants.H_index, :]),
        ("Latent_road (W/m^2)", road_meteo_data_temp[constants.L_index, :]),
        ("Surface_heatflux_road (W/m^2)", road_meteo_data_temp[constants.G_index, :]),
        ("Residual Energy (W/m^2)", road_meteo_data_temp[constants.E_index, :]),
        ("Energy correction (W/m^2)", road_meteo_data_temp[constants.E_correction_index, :]),
    ]
    for label, series in meteo_cols:
        _, v = _avg(date_num, series, i_min, i_max, av)
        cols[label] = v

    # Surface moisture
    water = g_road_data_temp[constants.water_index, :]
    ice_sum = np.sum(g_road_data_temp[constants.snow_ice_index, :], axis=0)
    evap = g_road_balance_data_temp[constants.water_index, constants.S_evap_index, :]
    drainage = g_road_balance_data_temp[constants.water_index, constants.S_drainage_index, :]

    for label, series in (
        ("Wetness_road (mm)", water),
        ("Ice_road (mm)", ice_sum),
        ("Evaporation (mm/hr)", evap),
        ("Drainage (mm/hr)", drainage),
        ("f_q_road", f_q_temp[constants.road_index, :]),
        ("f_q_brake", f_q_temp[constants.brake_index, :]),
        ("f_q_obs", f_q_obs_temp),
    ):
        _, v = _avg(date_num, series, i_min, i_max, av)
        cols[label] = v

    # Road maintenance
    salt2_str = input_activity.salt2_str
    maint_cols = [
        ("M_sanding (g/m^2)", activity_data_temp[constants.M_sanding_index, :]),
        ("M_salting(na) (g/m^2)", activity_data_temp[constants.M_salting_index[0], :]),
        (f"M_salting({salt2_str}) (g/m^2)", activity_data_temp[constants.M_salting_index[1], :]),
        ("Ploughing_road (0/1)", activity_data_temp[constants.t_ploughing_index, :]),
        ("Cleaning_road (0/1)", activity_data_temp[constants.t_cleaning_index, :]),
    ]
    for label, series in maint_cols:
        _, v = _avg(date_num, series, i_min, i_max, av)
        cols[label] = v

    # Mass balance and emissions (PM10)
    for label, series in (
        (
            "Total emissions PM10 (g/km/hr)",
            np.sum(
                E_road_data_temp[constants.dust_noexhaust_index, x, constants.E_total_index, :],
                axis=0,
            ),
        ),
        (
            "Direct emissions PM10 (g/km/hr)",
            np.sum(
                E_road_data_temp[constants.dust_noexhaust_index, x, constants.E_direct_index, :],
                axis=0,
            ),
        ),
        (
            "Suspended road emissions PM10 (g/km/hr)",
            np.sum(
                E_road_data_temp[constants.all_source_index, x, constants.E_suspension_index, :],
                axis=0,
            ),
        ),
        (
            "Road dust PM200 mass (g/m^2)",
            M_road_data_temp[constants.total_dust_index, x_load, :] * factor,
        ),
        (
            "Road sand PMall mass (g/m^2)",
            M_road_data_temp[constants.sand_index, constants.pm_all, :] * factor,
        ),
        (
            "Road sand PM200 mass (g/m^2)",
            M_road_data_temp[constants.sand_index, x_load, :] * factor,
        ),
        (
            "Road salt(na) mass (g/m^2)",
            M_road_data_temp[constants.salt_index[0], x_load, :] * factor,
        ),
        (
            f"Road salt({salt2_str}) mass (g/m^2)",
            M_road_data_temp[constants.salt_index[1], x_load, :] * factor,
        ),
        ("Road wear (g/km/hr)", WR_time_data_temp[constants.road_index, :]),
        ("Tyre wear (g/km/hr)", WR_time_data_temp[constants.tyre_index, :]),
        ("Brake wear (g/km/hr)", WR_time_data_temp[constants.brake_index, :]),
        (
            "Road dust production (g/km/hr)",
            M_road_balance_data_temp[
                constants.total_dust_index, x_load, constants.P_dusttotal_index, :
            ],
        ),
        (
            "Road dust sink (g/km/hr)",
            M_road_balance_data_temp[
                constants.total_dust_index, x_load, constants.S_dusttotal_index, :
            ],
        ),
    ):
        _, v = _avg(date_num, series, i_min, i_max, av)
        cols[label] = v

    # Emission factors (divide by N_total)
    _, emis_total_pm10 = _avg(
        date_num,
        np.sum(
            E_road_data_temp[constants.all_source_index, x, constants.E_total_index, :],
            axis=0,
        ),
        i_min,
        i_max,
        av,
    )
    _, n_total = _avg(date_num, traffic_data_temp[constants.N_total_index, :], i_min, i_max, av)
    with np.errstate(divide="ignore", invalid="ignore"):
        ef_pm10 = np.where(n_total != 0, emis_total_pm10 / n_total, np.nan)
    cols["Emission factor PM10 (g/km/veh)"] = ef_pm10

    # PM2.5 emissions
    for label, series in (
        (
            "Total emissions PM2.5 (g/km/hr)",
            np.sum(
                E_road_data_temp[constants.dust_noexhaust_index, x2, constants.E_total_index, :],
                axis=0,
            ),
        ),
        (
            "Direct emissions PM2.5 (g/km/hr)",
            np.sum(
                E_road_data_temp[constants.dust_noexhaust_index, x2, constants.E_direct_index, :],
                axis=0,
            ),
        ),
        (
            "Suspended road emissions PM2.5 (g/km/hr)",
            np.sum(
                E_road_data_temp[constants.all_source_index, x2, constants.E_suspension_index, :],
                axis=0,
            ),
        ),
    ):
        _, v = _avg(date_num, series, i_min, i_max, av)
        cols[label] = v

    _, emis_total_pm25 = _avg(
        date_num,
        np.sum(
            E_road_data_temp[constants.all_source_index, x2, constants.E_total_index, :],
            axis=0,
        ),
        i_min,
        i_max,
        av,
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        ef_pm25 = np.where(n_total != 0, emis_total_pm25 / n_total, np.nan)
    cols["Emission factor PM2.5 (g/km/veh)"] = ef_pm25

    # Concentrations
    def _avg_clean(series: np.ndarray) -> np.ndarray:
        s = series.copy().astype(float)
        s[s == nodata] = np.nan
        _, v = _avg(date_num, s, i_min, i_max, av)
        return v

    cols["PM10_obs net (ug/m^3)"] = _avg_clean(airquality_data.PM_obs_net[constants.pm_10, :])
    cols["PM10_obs bg (ug/m^3)"] = _avg_clean(airquality_data.PM_obs_bg[constants.pm_10, :])

    def _sum_pos(series: np.ndarray) -> np.ndarray:
        s = series.copy().astype(float)
        s[s < 0] = np.nan
        _, v = _avg(date_num, s, i_min, i_max, av)
        return v

    cols["PM10 mod total (ug/m^3)"] = _sum_pos(
        np.sum(
            C_data_temp[constants.all_source_noexhaust_index, x, constants.C_total_index, :],
            axis=0,
        )
    )
    cols["PM10 mod total+ep (ug/m^3)"] = _sum_pos(
        np.sum(C_data_temp[constants.all_source_index, x, constants.C_total_index, :], axis=0)
    )
    cols["PM10 mod dust total (ug/m^3)"] = _sum_pos(
        np.sum(C_data_temp[constants.dust_noexhaust_index, x, constants.C_total_index, :], axis=0)
    )
    cols["PM10 mod roadwear (ug/m^3)"] = _avg_clean(C_data_temp[constants.road_index, x, constants.C_total_index, :])
    cols["PM10 mod tyrewear (ug/m^3)"] = _avg_clean(C_data_temp[constants.tyre_index, x, constants.C_total_index, :])
    cols["PM10 mod brakewear (ug/m^3)"] = _avg_clean(C_data_temp[constants.brake_index, x, constants.C_total_index, :])
    cols["PM10 mod salt(na) (ug/m^3)"] = _avg_clean(C_data_temp[constants.salt_index[0], x, constants.C_total_index, :])
    cols[f"PM10 mod salt({salt2_str}) (ug/m^3)"] = _avg_clean(
        C_data_temp[constants.salt_index[1], x, constants.C_total_index, :]
    )
    cols["PM10 mod sand (ug/m^3)"] = _avg_clean(C_data_temp[constants.sand_index, x, constants.C_total_index, :])
    cols["PM10 mod exhaust (ug/m^3)"] = _avg_clean(C_data_temp[constants.exhaust_index, x, constants.C_total_index, :])

    # PM2.5 concentrations
    cols["PM25_obs net (ug/m^3)"] = _avg_clean(airquality_data.PM_obs_net[constants.pm_25, :])
    cols["PM25_obs bg (ug/m^3)"] = _avg_clean(airquality_data.PM_obs_bg[constants.pm_25, :])
    cols["PM25 mod total (ug/m^3)"] = _sum_pos(
        np.sum(
            C_data_temp[constants.all_source_noexhaust_index, x2, constants.C_total_index, :],
            axis=0,
        )
    )
    cols["PM25 mod total+ep (ug/m^3)"] = _sum_pos(
        np.sum(C_data_temp[constants.all_source_index, x2, constants.C_total_index, :], axis=0)
    )
    cols["PM25 mod dust total (ug/m^3)"] = _sum_pos(
        np.sum(C_data_temp[constants.dust_noexhaust_index, x2, constants.C_total_index, :], axis=0)
    )
    cols["PM25 mod roadwear (ug/m^3)"] = _avg_clean(C_data_temp[constants.road_index, x2, constants.C_total_index, :])
    cols["PM25 mod tyrewear (ug/m^3)"] = _avg_clean(C_data_temp[constants.tyre_index, x2, constants.C_total_index, :])
    cols["PM25 mod brakewear (ug/m^3)"] = _avg_clean(C_data_temp[constants.brake_index, x2, constants.C_total_index, :])
    cols["PM25 mod salt(na) (ug/m^3)"] = _avg_clean(C_data_temp[constants.salt_index[0], x2, constants.C_total_index, :])
    cols[f"PM25 mod salt({salt2_str}) (ug/m^3)"] = _avg_clean(
        C_data_temp[constants.salt_index[1], x2, constants.C_total_index, :]
    )
    cols["PM25 mod sand (ug/m^3)"] = _avg_clean(C_data_temp[constants.sand_index, x2, constants.C_total_index, :])
    cols["PM25 mod exhaust (ug/m^3)"] = _avg_clean(C_data_temp[constants.exhaust_index, x2, constants.C_total_index, :])

    return pd.DataFrame(cols)


