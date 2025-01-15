# Copyright (C) [2024] [Bo Zhou]
#
# This file is part of CSGtom project.
#
# CSGtom is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# CSGtom is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with CSGtom.  If not, see <https://www.gnu.org/licenses/>.

"""
Agricultural Biosystems Engineering Group, WUR
Institute of Urban Agriculture, Chinese Academy of Agricultural Sciences
@author: Bo Zhou （周波）
"""
import numpy as np
import copy

def parameters():
    p = {
        ###TODO: CSG shape parameters

        # CSG shape parameters (1) - first option---------#(1) first priority
        'A':np.array([1.7,4.6]),
        'B':np.array([0,3.3]),
        'C':np.array([0,0]),
        'D':np.array([8.5,0]),
        # -------------------------------------------------#
        # CSG shape parameters (2) - second option--------#(2)
        'width_indoorfloor': 8.5,  # CD
        # three of the four following parameters are enough
        'height_CSG': 4.6,      # A[1]
        'height_wall':3.3,      # B[1]
        'width_nroof':2.14,     # AB
        'angle_nroof':0.65,     # rad
        #--------------------------------------------------#

        ## curve of south cover
        'curveType': 'polyline', # 'polyline', 'ellipse', 'parabola', 'circle', 'manualfunc'-------'polyline' has the priority
        # in case of that: 'curveType': 'manualfunc'. this is example of manualfunc of the cover
        'curvefunc_gx': lambda x: ((4.600000 - -2.726087) ** 2 - (x - 1.700000) ** 2) ** 0.5 + -2.726087,
        # in case of that: 'curveType': 'polyline',
        'cover_rW': np.array([0.2, 0.16, 0.12, 0.14, 0.15, 0.03]),      #  if curveTpye='polyline' - from wall to south
        'cover_rH': np.array([0.09, 0.12, 0.1, 0.15, 0.28, 0.26]),      # if curveTpye='polyline'  - from wall to south
        'polyline_num':6,                  #default value is 100                    # if curveTpye='ellipse', 'parabola', 'circle', 'manualfunc'. This is the number of polylines for curve funciton

        # windows
        'r_net':0.6,        # the vent area ratio of pest net
        'Atop_vent':0.07,   # Maximum top ventilation vertical area [m2 per meter length of CSG]
        'Aside_vent':1.2,   # Maximum side ventilation vertical area[m2 per meter length of CSG]
        'htop':4.38,        # vertical height of middle point of top vent [m]
        'hside':0.5,        # vertical height of middle point of side vent [m]

        # material of wall, roof, cover, floor and layers
        # 1. wall
        'thickness_wall':np.array([0.02,0.06,0.12,0.18,0.12,0.06,0.02]),    # thickness of different layers of wall from internal to external [m]
        'density_wall':  np.array([1900,1900,1900,850,1900,1900,1900]),     # density of different layers of wall from internal to external [kg/m3]
        'capacity_wall': np.array([1.1,1.1,1.1,1.1,1.1,1.1,1.1]),           # heat capacity of different layers of wall from internal to external [kJ/kg/k]
        'heatcond_wall': np.array([0.75,0.75,0.75,0.05,0.75,0.75,0.75]),    # heat conductive of different layers of wall from internal to external [W/m/k]
        'FIRtran_wall':  np.array([0,0,0,0,0,0,0]),
        'FIRabs_wall':   np.array([0.93,0,0,0,0,0,0.93]),
        'NIRtran_wall':  np.array([0,0,0,0,0,0,0]),
        'NIRabs_wall':   np.array([0.6,0,0,0,0,0,0.6]),
        'PARtran_wall':  np.array([0,0,0,0,0,0,0]),
        'PARabs_wall':   np.array([0.25,0,0,0,0,0,0.25]),
        'difNIRtran_wall': np.array([0,0,0,0,0,0,0]),
        'difNIRabs_wall':  np.array([0.6, 0, 0, 0, 0, 0, 0.6]),
        'difPARtran_wall': np.array([0, 0, 0, 0, 0, 0, 0]),
        'difPARabs_wall':  np.array([0.25, 0, 0, 0, 0, 0, 0.25]),

        #2. soil
        'thickness_soil': np.array([0.01, 0.03, 0.03, 0.02, 0.23, 0.5]), # thickness of different layers of soil from internal to external [m]
        'density_soil': np.array([2050, 2050, 2050, 2050, 2050, 2050]),# density of different layers of soil from internal to external [kg/m3]
        'capacity_soil': np.array([1.01, 1.01, 1.01, 1.01, 1.01, 1.01]),# heat capacity of different layers of soil from internal to external [kJ/kg/k]
        'heatcond_soil': np.array([0.6, 0.6, 0.6, 0.6, 0.6, 0.6]),# heat conductive of different layers of soil from internal to external [W/m/k]
        'FIRtran_soil': np.array([0, 0, 0, 0, 0, 0]),
        'FIRabs_soil': np.array([0.96, 0, 0, 0, 0, 0]),
        'NIRtran_soil': np.array([0, 0, 0, 0, 0, 0]),
        'NIRabs_soil': np.array([0.33, 0, 0, 0, 0, 0]),
        'PARtran_soil': np.array([0, 0, 0, 0, 0, 0]),
        'PARabs_soil': np.array([0.33, 0, 0, 0, 0, 0]),
        'difNIRtran_soil': np.array([0, 0, 0, 0, 0, 0]),
        'difNIRabs_soil': np.array([0.33, 0, 0, 0, 0, 0]),
        'difPARtran_soil': np.array([0, 0, 0, 0, 0, 0]),
        'difPARabs_soil': np.array([0.33, 0, 0, 0, 0, 0]),

        #3. canopy
        'density_canopy':1,
        'capacity_canopy':1.2,
        'thickness_canopy':1,
        'K_par':0.7,   # extinction coefficient of the canopy for PAR
        'K_fir':0.94,   # extinction coefficient of the canopy for FIR
        'K_nir':0.27,   # extinction coefficient of the canopy for NIR

        #4. north roof
        'thickness_roof': np.array([0.08, 0.08, 0.14]),# thickness of different layers of roof from internal to external [m]
        'density_roof': np.array([900,900,900]), # density of different layers of roof from internal to external [kg/m3]
        'capacity_roof': np.array([2.3,2.3,2.3]),# heat capacity of different layers of roof from internal to external [kJ/kg/k]
        'heatcond_roof': np.array([0.42, 0.42, 0.42]),# heat conductive of different layers of roof from internal to external [W/m/k]
        'FIRtran_roof': np.array([0, 0, 0]),
        'FIRabs_roof': np.array([0.91, 0, 0.91]),
        'NIRtran_roof': np.array([0, 0, 0]),
        'NIRabs_roof': np.array([0.6, 0, 0.6]),
        'PARtran_roof': np.array([0, 0, 0]),
        'PARabs_roof': np.array([0.25, 0, 0.25]),
        'difNIRtran_roof': np.array([0, 0, 0]),
        'difNIRabs_roof': np.array([0.6, 0, 0.6]),
        'difPARtran_roof': np.array([0, 0,  0]),
        'difPARabs_roof': np.array([0.25, 0, 0.25]),

        #5. cover
        'density_cover':900,
        'capacity_cover':300,
        'thickness_cover':0.0001,
        'FIRtran_cover':0.1,
        'FIRabs_cover':0.9,
        'diftran':0.88,
        'heatcond_cover':0.42,

        #6. thermal blanket
        'thickness_blanket': np.array([0.01, 0.01]),  # thickness of different layers of blanket from internal to external [m]
        'density_blanket': np.array([107.8,107.8]),     # density of different layers of blanket from internal to external [kg/m3]
        'capacity_blanket': np.array([0.82,0.82]),          # heat capacity of different layers of blanket from internal to external [kJ/kg/k]
        'heatcond_blanket': np.array([0.03,0.03]),   # heat conductive of different layers of blanket from internal to external [W/m/k]
        'FIRtran_blanket': np.array([0, 0]),
        'FIRabs_blanket': np.array([0.9, 0.9]),
        'NIRtran_blanket': np.array([0, 0]),
        'NIRabs_blanket': np.array([0.6, 0.]),
        'PARtran_blanket': np.array([0, 0]),
        'PARabs_blanket': np.array([0.25, 0.]),
        'difNIRtran_blanket': np.array([0, 0]),
        'difNIRabs_blanket': np.array([0.6, 0.]),
        'difPARtran_blanket': np.array([0, 0]),
        'difPARabs_blanket': np.array([0.25, 0.]),

        #7. air
        'capacity_air': 1.0064,  # [kJ kg-1 K-1] Specific heat capacity of air at 25 [C]
        'density_air': 1.3,      # [ka/m3]
        'FIRabs_air':1,
        #8. CSG structure ratio on the south cover
        'R_glb_air':0.05,
        'GLB_abs_str':1,

        ###TODO: CSG ourdoor conditions

        'T_soilbound':20,   # boundry soil temperature
        'Water_con':0.1,    # soil water content [%]
        'C_CO2out':400,     # outdoor CO2 [ppm]
        'Angle_CSG':90,     # the greenhouse orientation [deg]
        'outdoorDataFileURL': '../data/example_data.xls',

        ###TODO: CSG-tomato yield model constant parameters
        #
        'r':0.8,        # the ratio of direct light of global radiation
        'r_PAR':0.5,    # the ratio of PAR among global radiation
        'Sigma':5.67E-8,    #Stefan Boltzmann constant
        'g':9.8,        # acceleration of gravity [m/s2]
        'capacity_water':4.2, # heat capacity of water in soil [J/K]

        # natural ventilation parameters
        'Cd':0.75,  # discharge coefficient which depends on the greenhouse shape.
        'Cw':0.09,  # Global wind pressure coefficient which depends on the greenhouse shape
        'Cleakage':5E-4,    # the leakage coefficient which depends on the greenhouse type ---- (1)first priority
        'air_leach':1.25,   # the air leaching/infiltration ration                [h-1]    ---- (2)second

        # vapour transpiration
        'Mwater':18,        #  Molar mass of water    [kg/kmol]
        'R':8.314E3,        #  molar gas constant     [J/kmolK]
        'gam':65.8,         #  psychometric constant   [Pa/K]
        'Lat':2.45E6,       #  latent heat of evaporation   [J/kg water]
        'rb':275,           #  boundary layer resistance of the canopy for vapour transport   [s/m]
        'rsmin':82,         #  minimum canopy resistance [s/m]
        'Cevap1':4.3,
        'Cevap2':0.54,
        'Cevap3_day':6.1E-7,
        'Cevap3_night':1.1E-11,
        'Cevap4_day':4.3E-6,
        'Cevap4_night':5.2E-6,
        'Cevap5_day':2.3E-2,
        'Cevap5_night':0.5E-2,
        'T_min_day':24.5,
        'T_min_night':33.6,
        'r1':275,
        'Cst':1,            # stress factor for vapour transpiration
        'Cst_con':0.5,      # stress factor for vapour condensation on the cover

        # direct light transmission
        'n':1.5,    # reflection index of plastic (optical property of the material)
        'C_abs':450,#2000,  # power absorption coefficient [m-1]
        'sim_t_tran':600,

        # heat exchange coefficient of convective
        'cin':7.2, # Convective heat exchange parameter between greenhouse elements and inside air    [W m-2 K-1]
        'cout1':2.8,# Convective heat exchange parameter between greenhouse elements and outside air   [W m-2 K-1]
        'cout2':1.2,# [J m-3 K-1]
        'cout3':1,  # []
        'c_can_air':5,
        # CO2
        'MCH2O':30,     #  Molar mass of CH2O    [kg/kmol]
        'MCO2':44,      #  Molar mass of CO2    [kg/kmol]
        # PLANT MODEL
        # PhotoSynthesis Buffer
        'LAI_Max':3,#2.5,  # [m2 {leaf} m^{-2}] Maximal value of LAI
        'THETA':0.7,  #
        'eta_ppm_mgm3':1.804,  #  Unit Transformation for CO2 Concentration  [mg/m3  per ppm]
        # CarboHydrate Buffer
        # Growth Inhibition (24 hour mean)
        'T24_S1':1.1587,
        'T24_b1':15,
        'T24_S2':-1.3904,
        'T24_b2':24.5,
        # Growth Inhibition (instanteneous)
        'Tinst_S1':0.869,
        'Tinst_b1':10,
        'Tinst_S2':-0.5793,
        'Tinst_b2':34,
        'Tsum_needed':1035,  # degreedays needed for full partioning to generative parts
        # Potential Growth
        'rg_fruit':0.328,
        'rg_leaf':0.0950,
        'rg_stemroot':0.0742,
        # Growth respiration coefficients based upon Heuvelink Phd, page 238
        'cfruit_g':0.27,
        'cleaf_g':0.28,
        'cstemroot_g':0.3,
        # Maintenance Respiration
        'NrBox':5,  # [-] Number of development stages
        'cfruit_m':1.16E-7,
        'cleaf_m':3.47E-7,
        'cstemroot_m':1.47E-7,
        'Resp_fac':1,    # (1 -exp(-f_RGR*RGR));
        'Q10m':2,           #  [-] Temperature Effect on Maintenance Respiration
        # Leaf Harvest
        'SLA':2.66E-5,  #  [m2 {leaf} mg^{-1} {CH2O}] Specific Leaf Area Index

        # simulation period
        'StartTime':'2017-09-01T00:00',   # '%Y-%m-%dT%H:%M' - year-month-day T hour: min
        'EndTime':  '2017-09-02T23:50',     # '%Y-%m-%dT%H:%M' - year-month-day T hour: min
        'dtsim': 600,   # [second] simulation result time difference
        'dt':30, # [second] time difference between euler simulation time

        # location
        'longitude':115.97,
        'latitude':39.62,
        'Hourcircle':120,

        # initial value
        'StateVariable': ['T_air','VP','CO2','LAI','T_floor','T_soil1','T_soil2','T_soil3','T_soil4','T_soil5','T_can','T_wali','T_wal1',\
                          'T_wal2','T_wal3','T_wal4','T_wal5','T_wale','T_rofi','T_rof1','T_rofe','T_cov','T_blanki','T_blanke',\
                          'T_sum','T_can24','C_buf','C_stemroot','C_leaf','C_har','C_fruit1','C_fruit2','C_fruit3','C_fruit4','C_fruit5',\
                          'N_fruit1','N_fruit2','N_fruit3','N_fruit4','N_fruit5'],
        'InitialValues': np.array([20, 2400,400, 0.5,21,20,20,20,20,20,20,25,23,\
                          20,20,20,20,19,21,20,19,18,5,5,\
                          -1400,25,10000,11278,11278,0,0.1,0.1,0.1,0.1,0.1,\
                          1,1,1,1,1]),
        # control of ventilation and thermal blanket
        'ctl_vent_type':'timebasedControl',          # 'proportionalControl', 'timebasedControl'
        'ctl_blank_type':'timebasedControl',         # 'proportionalControl', 'timebasedControl'

        # when 'proportionalControl', the four parameters will be asked
        'SP_blank_rad': 100,  # below which outdoor radiation the thermal blanket is planned to be used [W/m2]
        'SP_blank_tem':14,          # below which temperature the thermal blanket is used           [oC]
        'SP_ventTop':28,
        'SP_ventTopBot':32,

        # when 'timebasedControl', the following parameters will be asked
        'UFileURL': '../data/example_u.xls',

        # outdoor weather parameter for SA
        'SA_tem':1,
        'SA_RH':1,
        'SA_wind':1,
        'SA_rad':1,
        'SA_CO2':1,

        # p file name
        'name_p': 'example1',
        'MC':12  #  Molar mass of C    [kg/kmol]
    }
    return p

def SAparameter(p):
    p_sa = copy.deepcopy(p)
    del p_sa['A']
    del p_sa['B']
    del p_sa['C']
    del p_sa['D']
    del p_sa['angle_nroof']
    del p_sa['curveType']
    del p_sa['curvefunc_gx']
    del p_sa['polyline_num']
    del p_sa['StartTime']
    del p_sa['EndTime']
    del p_sa['StateVariable']
    del p_sa['InitialValues']
    del p_sa['ctl_vent_type']
    del p_sa['ctl_blank_type']
    del p_sa['NrBox']
    del p_sa['sim_t_tran']
    del p_sa['outdoorDataFileURL']
    return p_sa


