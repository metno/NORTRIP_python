"""
NORTRIP model constants package

Fixed indexing to be 0-indexed for Python compatibility
"""

# Number of roads. To be set by user but set here anyway
n_roads = 1

# Vehicle classes
he = 0
li = 1
num_veh = 2

# Tyre type
st = 0
wi = 1
su = 2
num_tyre = 3

# Salt type
na = 0
mg = 1
cma = 2
ca = 3
pfo = 4

# Moisture type
water_index = 0
snow_index = 1
ice_index = 2
rime_index = 3
frost_index = 4
num_moisture = 3
snow_ice_index = [snow_index, ice_index]

# Size fraction index
pm_all = 0
pm_200 = 1
pm_10 = 2
pm_25 = 3
num_size = 4
pm_course = 4
pm_sus = [pm_200, pm_10, pm_25]

# Source index
road_index = 0
tyre_index = 1
brake_index = 2
sand_index = 3
depo_index = 4
fugitive_index = 5
exhaust_index = 6
salt_index = [7, 8]  # salt_index(0)=7; salt_index(1)=8
total_dust_index = 9
crushing_index = 10
abrasion_index = 11
num_wear = 3
num_dust = 7
num_salt = 2
num_source = 9
num_source_all = 10
num_source_all_extra = 12
dust_index = list(range(0, 7))  # 0:6
dust_noexhaust_index = list(range(0, 6))  # 0:5
wear_index = list(range(0, 3))  # 0:2
all_source_index = list(range(0, num_source))
all_source_noexhaust_index = dust_noexhaust_index + salt_index

# Strings defining the plot and save averaging output types
av_str = ["hour", "day", "dailycycle", "halfday", "weekdays", "dayrun", "week", "month"]

# Date data indexing
year_index = 0
month_index = 1
day_index = 2
hour_index = 3
minute_index = 4
datenum_index = 5
num_date_index = 6

# Traffic data indexing
N_total_index = 0
N_v_index = [1, 2]  # [he=1, li=2]
N_t_v_index = {
    (st, he): 3,
    (wi, he): 4,
    (su, he): 5,
    (st, li): 6,
    (wi, li): 7,
    (su, li): 8,
}
V_veh_index = [9, 10]  # [he=9, li=10]
num_traffic_index = 11

# Meteo indexes
T_a_index = 0
T2_a_index = 1
FF_index = 2
DD_index = 3
RH_index = 4
Rain_precip_index = 5
Snow_precip_index = 6
short_rad_in_index = 7
long_rad_in_index = 8
short_rad_in_clearsky_index = 9
cloud_cover_index = 10
road_temperature_obs_input_index = 11
road_wetness_obs_input_index = 12
pressure_index = 13
T_dewpoint_index = 14
T_sub_input_index = 15
num_meteo_index = 16

# Activity indexes
M_sanding_index = 0
t_ploughing_index = 1
t_cleaning_index = 2
g_road_wetting_index = 3
M_salting_index = [4, 5]  # [4, 5] for index 0 and 1
M_fugitive_index = 6
num_activity_index = 7

# Dust balance indexes
S_dusttotal_index = 0
P_dusttotal_index = 1
P_wear_index = 2
S_dustspray_index = 3
P_dustspray_index = 4
S_dustdrainage_index = 5
S_suspension_index = 6
S_windblown_index = 7
S_cleaning_index = 8
P_cleaning_index = 9
S_dustploughing_index = 10
P_crushing_index = 11
S_crushing_index = 12
P_abrasion_index = 13
P_depo_index = 14
num_dustbalance = 15

# Salt solution indexes
RH_salt_index = 0
melt_temperature_salt_index = 1
dissolved_ratio_index = 2
num_saltdata = 3

# Dust process emission indexes
E_direct_index = 0
E_suspension_index = 1
E_windblown_index = 2
E_total_index = 3

# Dust process concentration indexes
C_direct_index = 0
C_suspension_index = 1
C_windblown_index = 2
C_total_index = 3
num_process = 4

# Road meteorological data
T_s_index = 0
T_melt_index = 1
r_aero_index = 2
r_aero_notraffic_index = 3
RH_s_index = 4
RH_salt_final_index = 5
L_index = 6
H_index = 7
G_index = 8
G_sub_index = 9
G_freeze_index = 10
G_melt_index = 11
evap_index = 12
evap_pot_index = 13
rad_net_index = 14
short_rad_net_index = 15
long_rad_net_index = 16
long_rad_out_index = 17
H_traffic_index = 18
road_temperature_obs_index = 19
road_wetness_obs_index = 20
T_sub_index = 21
short_rad_net_clearsky_index = 22
T_s_dewpoint_index = 23
E_index = 24
E_correction_index = 25
num_road_meteo = 26

# Road moisture mass balance production and sink data
S_melt_index = 0
P_melt_index = 1
P_freeze_index = 2
S_freeze_index = 3
P_evap_index = 4
S_evap_index = 5
S_drainage_index = 6
S_spray_index = 7
R_spray_index = 8
P_spray_index = 9
S_total_index = 10
P_total_index = 11
P_precip_index = 12
P_roadwetting_index = 13
S_drainage_tau_index = 14
R_drainage_index = 15
num_moistbalance = 16

# Air quality indexes
PM10_obs_index = 0
PM10_bg_index = 1
PM10_net_index = 2
PM25_obs_index = 3
PM25_bg_index = 4
PM25_net_index = 5
NOX_obs_index = 6
NOX_bg_index = 7
NOX_net_index = 8
NOX_emis_index = 9
EP_emis_index = 10
f_conc_index = 11
num_airquality_index = 12

# PM observation index mappings
PM_obs_index = {pm_10: PM10_obs_index, pm_25: PM25_obs_index}
PM_bg_index = {pm_10: PM10_bg_index, pm_25: PM25_bg_index}
PM_net_index = {pm_10: PM10_net_index, pm_25: PM25_net_index}

# Efficiency indexes
ploughing_eff_index = 0
cleaning_eff_index = 1
drainage_eff_index = 2
spraying_eff_index = 3

# Number of 'tracks' on the road
alltrack_type = 0
outtrack_type = 1
intrack_type = 2
shoulder_type = 3
kerb_type = 4
num_track_max = 5
