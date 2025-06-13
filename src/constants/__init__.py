"""
NORTRIP model constants package

TODO: Fix the indexing for the Python to be 0 indexed
"""

# Number of roads. To be set by user but set here anyway
n_roads = 1

# Vehicle classes
# TODO: Are these indexed?
he = 1
li = 2
num_veh = 2

# Tyre type
st = 1
wi = 2
su = 3
num_tyre = 3

# Salt type
na = 1
mg = 2
cma = 3
ca = 4
pfo = 5

# Moisture type
water_index = 1
snow_index = 2
ice_index = 3
rime_index = 4
frost_index = 5
num_moisture = 3
snow_ice_index = [snow_index, ice_index]

# Size fraction index
pm_all = 1
pm_200 = 2
pm_10 = 3
pm_25 = 4
num_size = 4
pm_course = 5
pm_sus = [pm_200, pm_10, pm_25]

# Source index
road_index = 1
tyre_index = 2
brake_index = 3
sand_index = 4
depo_index = 5
fugitive_index = 6
exhaust_index = 7
salt_index = [8, 9]  # salt_index(1)=8; salt_index(2)=9
total_dust_index = 10
crushing_index = 11
abrasion_index = 12
num_wear = 3
num_dust = 7
num_salt = 2
num_source = 9
num_source_all = 10
num_source_all_extra = 12
dust_index = list(range(1, 8))  # 1:7
dust_noexhaust_index = list(range(1, 7))  # 1:6
wear_index = list(range(1, 4))  # 1:3
all_source_index = list(range(1, num_source + 1))
all_source_noexhaust_index = dust_noexhaust_index + salt_index

# Strings defining the plot and save averaging output types
av_str = ["hour", "day", "dailycycle", "halfday", "weekdays", "dayrun", "week", "month"]

# Date data indexing
year_index = 1
month_index = 2
day_index = 3
hour_index = 4
minute_index = 5
datenum_index = 6
num_date_index = 6

# Traffic data indexing
N_total_index = 1
N_v_index = [2, 3]  # [0, he=2, li=3] (0-indexed for unused)
N_t_v_index = {
    (st, he): 4,
    (wi, he): 5,
    (su, he): 6,
    (st, li): 7,
    (wi, li): 8,
    (su, li): 9,
}
V_veh_index = [10, 11]  # [0, he=10, li=11]
num_traffic_index = 11

# Meteo indexes
T_a_index = 1
T2_a_index = 2
FF_index = 3
DD_index = 4
RH_index = 5
Rain_precip_index = 6
Snow_precip_index = 7
short_rad_in_index = 8
long_rad_in_index = 9
short_rad_in_clearsky_index = 10
cloud_cover_index = 11
road_temperature_obs_input_index = 12
road_wetness_obs_input_index = 13
pressure_index = 14
T_dewpoint_index = 15
T_sub_input_index = 16
num_meteo_index = 16

# Activity indexes
M_sanding_index = 1
t_ploughing_index = 2
t_cleaning_index = 3
g_road_wetting_index = 4
M_salting_index = [0, 5, 6]  # [0, 5, 6] for index 1 and 2
M_fugitive_index = 7
num_activity_index = 7

# Dust balance indexes
S_dusttotal_index = 1
P_dusttotal_index = 2
P_wear_index = 3
S_dustspray_index = 4
P_dustspray_index = 5
S_dustdrainage_index = 6
S_suspension_index = 7
S_windblown_index = 8
S_cleaning_index = 9
P_cleaning_index = 10
S_dustploughing_index = 11
P_crushing_index = 12
S_crushing_index = 13
P_abrasion_index = 14
P_depo_index = 15
num_dustbalance = 15

# Salt solution indexes
RH_salt_index = 1
melt_temperature_salt_index = 2
dissolved_ratio_index = 3
num_saltdata = 3

# Dust process emission indexes
E_direct_index = 1
E_suspension_index = 2
E_windblown_index = 3
E_total_index = 4

# Dust process concentration indexes
C_direct_index = 1
C_suspension_index = 2
C_windblown_index = 3
C_total_index = 4
num_process = 4

# Road meteorological data
T_s_index = 1
T_melt_index = 2
r_aero_index = 3
r_aero_notraffic_index = 4
RH_s_index = 5
RH_salt_final_index = 6
L_index = 7
H_index = 8
G_index = 9
G_sub_index = 10
G_freeze_index = 11
G_melt_index = 12
evap_index = 13
evap_pot_index = 14
rad_net_index = 15
short_rad_net_index = 16
long_rad_net_index = 17
long_rad_out_index = 18
H_traffic_index = 19
road_temperature_obs_index = 20
road_wetness_obs_index = 21
T_sub_index = 22
short_rad_net_clearsky_index = 23
T_s_dewpoint_index = 24
E_index = 25
E_correction_index = 26
num_road_meteo = 26

# Road moisture mass balance production and sink data
S_melt_index = 1
P_melt_index = 2
P_freeze_index = 3
S_freeze_index = 4
P_evap_index = 5
S_evap_index = 6
S_drainage_index = 7
S_spray_index = 8
R_spray_index = 9
P_spray_index = 10
S_total_index = 11
P_total_index = 12
P_precip_index = 13
P_roadwetting_index = 14
S_drainage_tau_index = 15
R_drainage_index = 16
num_moistbalance = 16

# Air quality indexes
PM10_obs_index = 1
PM10_bg_index = 2
PM10_net_index = 3
PM25_obs_index = 4
PM25_bg_index = 5
PM25_net_index = 6
NOX_obs_index = 7
NOX_bg_index = 8
NOX_net_index = 9
NOX_emis_index = 10
EP_emis_index = 11
f_conc_index = 12
num_airquality_index = 12

# PM observation index mappings
# TODO: Check if these are correct
PM_obs_index = [0, 0, 0, PM10_obs_index, PM25_obs_index]  # pm_10=3, pm_25=4
PM_bg_index = [0, 0, 0, PM10_bg_index, PM25_bg_index]
PM_net_index = [0, 0, 0, PM10_net_index, PM25_net_index]

# Efficiency indexes
ploughing_eff_index = 1
cleaning_eff_index = 2
drainage_eff_index = 3
spraying_eff_index = 4

# Number of 'tracks' on the road
alltrack_type = 1
outtrack_type = 2
intrack_type = 3
shoulder_type = 4
kerb_type = 5
num_track_max = 5
