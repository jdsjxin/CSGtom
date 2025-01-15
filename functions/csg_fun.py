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
@authors: Bo Zhou（周波）

CSG functions
"""

import time
import numpy as np
import math
import xlrd
from scipy import interpolate


def solarAlt(t,Date,location):
    """

    Parameters
    ----------
    t: 1 size, simulation time
    Date: start time, Datetime 'year-month-dayTHour:Min', eg: '2017-09-01T18:00'
    location: [latitude,longitude,Hourcircle]

    Returns
    -------
    [solar_alt, Angle_time, solar_dec]

    """
    dtime = location[1]-location[2]
    startTime =time.strptime(Date,'%Y-%m-%dT%H:%M')
    day_num = np.mod(startTime.tm_yday+np.floor((startTime.tm_hour+startTime.tm_min/60+t/3600)/24),365)            #  day number
    gam = (day_num - 81) * 2 * math.pi / 365 # the year angle which is zero at the vernal equinox about March 21  [rad]
    dEqofT = 7.13 * np.cos(gam) + 1.84 * np.sin(gam) + 0.69 * np.cos(2 * gam) - 9.92 * np.sin(2 * gam)   # A fit curve for equation of time[min]
    solar_dec = 23.45 * np.sin(gam) * math.pi / 180 # solar declination [rad]
    Angle_time = (startTime.tm_hour + (t + dtime) / 3600 - dEqofT / 60 - 12) * 2 * math.pi / 24  # time angle [rad]
    solar_alt_sin = np.sin(location[0] * math.pi / 180) * np.sin(solar_dec) + np.cos(location[0] * math.pi / 180) * np.cos(solar_dec) * np.cos(Angle_time)
    solar_alt1 = math.asin(solar_alt_sin) # solar altitude           [rad]
    solar_alt = solar_alt1 * switch01(np.array([solar_alt1]), -1)
    angle_solar = np.array([solar_alt[0],Angle_time,solar_dec])
    return angle_solar

def directtran(angle_solar,angle_csg,cover,Material):
    """

    Parameters
    ----------
    angle_solar: [solar altitude,time angle, solar declination]
    angle_csg: greenhouse orientation [deg]
    cover: np.array([[slop(rad)], [length(m)]])
    Material: [reflection index, thickness(m),power absorption coefficient(/m)]

    Returns
    -------

    """
    solar_azimuth = math.asin(np.cos(angle_solar[2])*np.sin(angle_solar[1])*zerodivided0(np.array([1]), np.array([np.cos(angle_solar[0])])))  # solar azimuth relative to N-S [rad]
    angle_g = angle_csg * math.pi / 180  # greenhouse orientation to N-S [rad]
    angle_inci_cos = np.cos(angle_solar[0]) * np.sin(solar_azimuth + angle_g) * np.sin(cover[0,:])+np.sin(angle_solar[0]) * np.cos(cover[0,:])
    angle_inci=np.array([])
    for cosi in angle_inci_cos:
        angle_inci=np.append(angle_inci,math.acos(cosi))    # angle of incidence between normal(perpendicular) line [rad]
    c = (Material[0] ** 2 - np.sin(angle_inci)**2)**0.5  # Material(1) is reflection index of plastic (optical property of the material)
    Rpa_single = ((np.cos(angle_inci) - c) / (np.cos(angle_inci) + c))**2  # single reflection of parallel light
    Rpe_single = ((Material[0]**2 * np.cos(angle_inci) - c) / ( Material[0]**2 * np.cos(angle_inci) + c))**2  # single reflection of perpendicular light
    Pathway = Material[1] / (1 - (np.sin(angle_inci) / Material[0])**2)**0.5  # the optical pathway [m]
    absorp = np.exp(-2 * Material[2] * Pathway)  # the absorption of the material
    Rpa_mul = Rpa_single + Rpa_single * (1 - Rpa_single)**2 * absorp / (1 - Rpa_single**2 * absorp) # for multiple reflection with transmission through the pane-parallel
    Rpe_mul = Rpe_single + Rpe_single * (1 - Rpe_single)**2 * absorp / ( 1 - Rpe_single**2 * absorp)  # for multiple reflection with transmission through the pane-perpendicular
    Tpa = (1 - Rpa_mul)**2 * absorp**0.5 / (1 - Rpa_mul*absorp)  # transmission of the parallel light
    Tpe = (1 - Rpe_mul)**2 * absorp**0.5 / (1 - Rpe_mul*absorp)  # transmission of the perpendicular light
    Tpa[Tpa<0]=0
    Tpe[Tpe<0]=0
    tran_tot = (Tpa + Tpe) / 2*switch01(np.array([angle_solar[0]]), -1) # total transmission of light (assuming non polarzed light)
    # total direct light transmission of cover
    tran = np.sum(tran_tot * cover[1, :]*np.sin(math.pi - cover[0,:]-angle_solar[0])) / (np.sum(cover[1, :] * np.sin(math.pi - cover[0,:]-angle_solar[0])))
    return tran

def switch01(x,q):   # x-np.array
    k=1E16
    a=1/(1+np.exp(q*k*np.nan_to_num(x)))
    sw=np.empty(a.shape[0])
    for i in np.arange(0,a.shape[0],1):
        sw[i]=int(a[i])
    return sw
def switch02(x,q):    # x-np.array
    k=1E16
    a=1/(1+np.exp(q*k*np.nan_to_num(x)))+1
    sw=np.empty(a.shape[0])
    for i in np.arange(0,a.shape[0],1):
        sw[i]=round(a[i])-1
    return sw
def switch03(x,q):
    k = -10
    a=1/(1+np.exp(q*k*x))
    sw = a
    return sw
def vapourPsat(T):
    vpsat = 10**(2.7857+9.5*T/(265.5+T))*switch01(T,1)+10**(2.7857+7.5*T/(237.3+T))*switch01(T,-1)
    return vpsat
def dptem(tem,rh):    #tem,rh: np.array
    tem[tem==0]=0.001
    esat=vapourPsat(tem)
    e=esat*rh
    dp1 = -60.45+7.0322*np.log(e)+0.3700*np.log(e)**2
    dp2 = -35.957-1.8726*np.log(e)+1.1689*np.log(e)**2
    sw1 = abs(switch02(dp1,1)-switch01(dp2,-1)) #% incase dp1<=0 and dp2>0
    dp = (dp1*switch02(dp1,1)+dp2*switch01(dp2,-1))*sw1
    dp = (dp1+dp2)/2*switch02(sw1,1)+dp*switch01(sw1,-1) #%if sw1=0, dp = (dp1+dp2)/2
    return dp

def zerodivided0(a,b):
    """
    zerodivided0: a divided by b.
    if the value of b is 0, the answer is 0; else ans=a/b.
    Parameters
    ----------
    a
    b

    Returns
    -------

    """
    c=switch01(np.abs(b),-1)
    d=switch02(np.abs(b),1)
    div = a/(b+d)*c
    return div



def outdoorWeather_xls_st2end(p):
    """

    Parameters
    ----------
    p['outdoorDataFileURL']: excel url
    p['C_CO2out']: outdoor CO2 values, float

    Returns
    -------

    """
    data = xlrd.open_workbook(p['outdoorDataFileURL'])
    table = data.sheets()[0]
    data_array={}
    keylist =[]
    for rcol in range(table.ncols):
        data_array[table.cell_value(0,rcol)]=np.array([])
        keylist.append(table.cell_value(0,rcol))
    if 'CO2' not in data_array:
        if 'C_CO2out' in p:
            data_array['CO2']=np.ones(table.nrows-1)*p['C_CO2out']*p['SA_CO2']
        else:
            data_array['CO2'] = np.ones(table.nrows-1) * 400 # default ourdoor CO2 is 400 ppm
    else:
        for rown in range(1,table.nrows):
            data_array['CO2']=np.append(data_array['CO2'],table.cell_value(rown,keylist.index('CO2'))*p['SA_CO2'])
    data_array['Tsky']=np.array([])

    for rown in range(1,table.nrows):
        data_array['time'] = np.append(data_array['time'], table.cell_value(rown, keylist.index('time')))
        data_array['Tem'] = np.append(data_array['Tem'], table.cell_value(rown, keylist.index('Tem'))*p['SA_tem'])
        data_array['Rad'] = np.append(data_array['Rad'], table.cell_value(rown, keylist.index('Rad'))*p['SA_rad'])
        data_array['Wind'] = np.append(data_array['Wind'], table.cell_value(rown, keylist.index('Wind'))*p['SA_wind'])
        data_array['RH'] = np.append(data_array['RH'], table.cell_value(rown, keylist.index('RH'))*p['SA_RH'])
        data_array['realTime'] = np.append(data_array['realTime'], time.strftime("%Y-%m-%dT%H:%M",\
                                                                                 time.localtime((table.cell_value(rown, keylist.index('realTime'))-19-70*365)*86400-8*3600)))
        dewpointTem = dptem(np.array([table.cell_value(rown, keylist.index('Tem'))]),np.array([table.cell_value(rown, keylist.index('RH'))]))
        data_array['Tsky'] = np.append(data_array['Tsky'],(0.74+0.006*dewpointTem)**0.25*(table.cell_value(rown, keylist.index('Tem'))+273)-273)

    data_array['time'] = data_array['time'] - data_array['time'][list(data_array['realTime']).index(p['StartTime'])]
    # interpolate function
    data_array['f_Tem'] = interpolate.interp1d(data_array['time'],data_array['Tem'],kind='slinear',fill_value='extrapolate')
    data_array['f_Rad'] = interpolate.interp1d(data_array['time'], data_array['Rad'], kind='slinear',
                                               fill_value='extrapolate')
    data_array['f_Wind'] = interpolate.interp1d(data_array['time'], data_array['Wind'], kind='slinear',
                                               fill_value='extrapolate')
    data_array['f_RH'] = interpolate.interp1d(data_array['time'], data_array['RH'], kind='slinear',
                                               fill_value='extrapolate')
    data_array['f_Tsky'] = interpolate.interp1d(data_array['time'], data_array['Tsky'], kind='slinear',
                                               fill_value='extrapolate')
    data_array['f_CO2'] = interpolate.interp1d(data_array['time'], data_array['CO2'], kind='slinear',
                                               fill_value='extrapolate')

    return data_array

def ctl_csg1(p, D, d, Tair,_t,U):
    wind = d['f_Wind'](_t)
    Tout = d['f_Tem'](_t)
    rad_out = d['f_Rad'](_t)
    Atop_vent = p['Atop_vent']*p['r_net']
    Aside_vent = p['Aside_vent']*p['r_net']
    # Top and bottom vent
    vent_top_bot = p['Cd']/D.area_floor * ((Atop_vent*Aside_vent/(Atop_vent**2+Aside_vent**2)**0.5)**2*\
    (2*p['g']*(p['htop']-p['hside'])*(max((Tair-Tout),0))/(Tair/2+Tout/2+273.15))+(Atop_vent/2+Aside_vent/2)**2*p['Cw']*wind**2)**0.5
    # Top vent
    vent_top = Atop_vent * p['Cd'] / 2 / D.area_floor * (p['g'] * p['htop'] / 2 * (max((Tair - Tout), 0)) / ((Tair + Tout) / 2 + 273.15) + p['Cw'] * wind**2)**0.5
    # side vent
    vent_side = p['Cd'] / 2 / D.area_floor * Aside_vent * wind * p['Cw']**0.5
    # air leaching [ m3/ m2 s]
    if 'Cleakage' in p:
        Leaching_air = max(p['Cleakage']*0.25, p['Cleakage']*wind)
    else:
        Leaching_air = p['air_leach']*D.Vair/D.area_floor/3600
    # u vector [0 or 1]
    if p['ctl_vent_type'] == 'timebasedControl' and p['ctl_blank_type'] == 'timebasedControl':
        u_vent = U['u_vent'](_t)
        u_venttop = U['u_venttop'](_t)
        u_ventside = U['u_ventside'](_t)
        u_venttopbot = U['u_venttopbot'](_t)
        u_blanket = U['u_blanket'](_t)
    if p['ctl_blank_type'] == 'proportionalControl':
        u_blanket = switch01(np.array([p['SP_blank_tem'] - Tair]), -1)*switch01(np.array([rad_out - p['SP_blank_rad']]),1)
    if p['ctl_vent_type'] == 'proportionalControl':
        u_vent = switch01(np.array([Tair - p['SP_ventTop']]), -1) * (1 - u_blanket)
        u_venttopbot = switch01(np.array([Tair - p['SP_ventTopBot']]), -1)
        u_venttop = switch01(np.array([Tair - p['SP_ventTop']]), -1) * (1 - u_venttopbot)
        u_ventside = 0
    Vent = Leaching_air/(1+u_blanket) + (1 - u_blanket) * u_vent * (vent_top_bot * u_venttopbot + vent_side * u_ventside + vent_top * u_venttop)  # [m3 / m2 s]
    H_airvent = D.area_floor * Vent * (Tair - Tout) * p['capacity_air'] * p['density_air'] * 1000   # [W per meter length CSG]

    return u_blanket, u_vent, u_venttopbot, u_ventside, Vent, H_airvent
def conductive(d1,lamda1,t1,d2,lamda2,t2, area):
    HEC = 1/(0.5*d1/lamda1+0.5*d2/lamda2)*area*(t1-t2)
    return HEC

def vapour(p,D,T_air,VP_air,T_cov,T_can,T_so,T_out,RH_out,area_cov,area_flo,Rad_aboveCan_evap,C_CO2,Capd_air,LAI,Vent,ext_vp):
    VP_sat_cov = vapourPsat(np.array([T_cov]))
    VP_sat_can = vapourPsat(np.array([T_can]))
    VP_sat_so = vapourPsat(np.array([T_so]))
    VP_sat_out = vapourPsat(np.array([T_out]))
    VP_out = VP_sat_out *RH_out         # outside air vapour pressure [pa]
    Cvp_in = VP_air * p['Mwater']/(p['R']*(273.15+T_air))       # inside vapour concentration [kg/m3]
    Cvp_out = VP_out * p['Mwater']/(p['R']*(273.15+T_out))      # outside vapour concentration [kg/m3]

    # condensation from cover
    MV_cov = p['Cst_con'] * max(0, 6.4e-9 * p['cin']*area_cov/area_flo * (VP_air - VP_sat_cov))           #  the vapour exchange coefficient between air and cover   [kg/m2s]

    # transpiration from canopy
    Srs = switch03((Rad_aboveCan_evap-5),-1)  # switch function for night and day time
    Cevap3 = p['Cevap3_night'] * (1 - Srs) + p['Cevap3_day'] * Srs
    Cevap4 = p['Cevap4_night'] * (1 - Srs) + p['Cevap4_day'] * Srs
    Cevap5 = p['Cevap5_night'] * (1 - Srs) + p['Cevap5_day'] * Srs
    T_min = p['T_min_night'] * (1 - Srs) + p['T_min_day'] * Srs

    rf_R = (Rad_aboveCan_evap + p['Cevap1']) / (Rad_aboveCan_evap + p['Cevap2'])        #  resistance factor for high radiation levels
    rf_CO2 = (1 + Srs * Cevap3 * (C_CO2 - 200)**2) * switch01(np.array([C_CO2 - 1100]), 1) + 1.5 * (1 - switch01(np.array([C_CO2 - 1100]), 1))  # resistance factor for high CO2 levels
    rf_VP = 3.8 * switch02(1 + Cevap4 * (VP_sat_can - VP_air)**2 - 3.8, -1) + ( 1 + Cevap4 * (VP_sat_can - VP_air)**2) * switch01(1 + Cevap4 * (VP_sat_can - VP_air)**2 - 3.8,1) # resistance factor for large vapour pressure
    rf_T = 1 + Cevap5 * (T_can - T_min)**2 # resistance factor for temperature
    rs = p['rsmin'] * rf_R * rf_CO2 * rf_VP * rf_T # stomatal resistance of the canopy for vapour transport[s / m]
    VEC_canair = 2 * Capd_air / D.Vair * LAI / (p['Lat'] * p['gam'] * (p['rb'] + rs)) # vapour exchange coefficien between the canopy and air[kg Pa / s]
    dvp1 = max(VP_sat_can- VP_air, 0)
    MV_canair = max(0, p['Cst'] * VEC_canair * dvp1)      # canopy transpiration rate % [kg / m2 s]
    # condensation from canopy
    MV_can_cod =  max(0, 6.4e-9 * p['c_can_air']*LAI * (VP_air - VP_sat_can))
    MV_canair = MV_canair - MV_can_cod

    # evaporation from soil
    r2 = 3.5*p['Water_con']**-2.3  # the surface resistance of the bare soil [s/m]
    beta_soil = 1 / (1 + r2 / p['r1'])
    VEC_soilair = Capd_air / D.Vair / (p['Lat'] * p['gam']  * (p['r1'] + r2))  # vapour exchange coefficient between the soil and air  [kg Pa/s]
    dvp2 = max(VP_sat_so - VP_air, 0)
    MV_soil = max(0, beta_soil * VEC_soilair * dvp2)  # soil evaporation   %[kg/m2 s]

    # ventilation
    MV_vent = Cvp_out - Cvp_in        # vapour remove rate according to ventilation [kg/m3]

    #
    M = [MV_cov * D.area_floor, MV_canair * D.area_floor, MV_soil * D.area_floor, MV_vent * Vent * D.area_floor]
    dM_vp = -MV_cov * D.area_floor + MV_canair * D.area_floor + MV_soil * D.area_floor + MV_vent * Vent * D.area_floor + ext_vp
    L_air_cov = MV_cov * D.area_floor* p['Lat']
    L_air_can = -MV_canair * D.area_floor * p['Lat']
    L_air_so  = - MV_soil * D.area_floor * p['Lat']
    dVPair =  dM_vp / D.Vair * p['R'] * (T_air + 273.15) / p['Mwater']
    return dVPair, M, L_air_cov, L_air_can, L_air_so
def ctl_csg_pre(p):
    u_data = read_xls(p['UFileURL'])
    tsim = np.arange(0, time.mktime(time.strptime(p['EndTime'], "%Y-%m-%dT%H:%M")) - time.mktime(
        time.strptime(p['StartTime'], "%Y-%m-%dT%H:%M")), p['dt'])
    u_vent=[]
    u_venttop=[]
    u_ventside=[]
    u_venttopbot = []
    u_blanket = []
    for _t in tsim:
        u_t = time.localtime(time.mktime(time.strptime(p['StartTime'],'%Y-%m-%dT%H:%M'))+_t)
        u_vent.append((u_data['Vent'][u_data['DOY']==u_t.tm_yday])[0])
        u_venttop.append((u_data['Top'][u_data['DOY']==u_t.tm_yday]/(1+np.exp(10*(u_t.tm_hour+u_t.tm_min/60-u_data['Vent_start'][u_data['DOY']==u_t.tm_yday])*(u_t.tm_hour+u_t.tm_min/60-u_data['Vent_end'][u_data['DOY']==u_t.tm_yday]))))[0])
        u_ventside.append((u_data['Bottom'][u_data['DOY'] == u_t.tm_yday] / (1 + np.exp(10 * (u_t.tm_hour + u_t.tm_min / 60 - u_data['Vent_start'][u_data['DOY'] == u_t.tm_yday]) * (
                        u_t.tm_hour + u_t.tm_min / 60 - u_data['Vent_end'][u_data['DOY'] == u_t.tm_yday]))))[0])
        u_venttopbot.append((u_data['TopBot'][u_data['DOY'] == u_t.tm_yday] / (1 + np.exp(
                10 * (u_t.tm_hour + u_t.tm_min / 60 - u_data['Vent_start'][u_data['DOY'] == u_t.tm_yday]) * (
                            u_t.tm_hour + u_t.tm_min / 60 - u_data['Vent_end'][u_data['DOY'] == u_t.tm_yday]))))[0])

        u_blanket1 = (u_data['Blanket'][u_data['DOY'] == u_t.tm_yday]/(1+np.exp(-20*(u_t.tm_hour + u_t.tm_min / 60-u_data['Blanket_morning'][u_data['DOY'] == u_t.tm_yday])*(u_t.tm_hour + u_t.tm_min / 60-u_data['Blanket_afternoon'][u_data['DOY'] == u_t.tm_yday]))))[0]
        u_blanket.append(round(u_blanket1,1))
    U={'u_blanket':interpolate.interp1d(tsim, np.array(u_blanket), kind='slinear',fill_value='extrapolate'),
       'u_vent':interpolate.interp1d(tsim, np.array(u_vent), kind='slinear',fill_value='extrapolate'),
       'u_venttopbot':interpolate.interp1d(tsim, np.array(u_venttopbot), kind='slinear',fill_value='extrapolate'),
       'u_ventside':interpolate.interp1d(tsim, np.array(u_ventside), kind='slinear',fill_value='extrapolate'),
       'u_venttop':interpolate.interp1d(tsim, np.array(u_venttop), kind='slinear',fill_value='extrapolate')}
    return U

def read_xls(p):
    """

    Parameters
    ----------
    p: excel url

    Returns
    -------

    """
    data = xlrd.open_workbook(p)
    table = data.sheets()[0]
    data_array={}
    keylist =[]
    for rcol in range(table.ncols):
        data_array[table.cell_value(0,rcol)]=np.array([])
        keylist.append(table.cell_value(0,rcol))

    for rown in range(1,table.nrows):
        for k in keylist:
            data_array[k] = np.append(data_array[k], table.cell_value(rown, keylist.index(k)))

    return data_array