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
"""


import numpy as np

from classes.module import Module
from functions.integration import fcn_euler_forward
from functions import csg_shape, csg_fun
from functions import BVtomato_fun
from parameters import example
import time

class CSG_Climate(Module):
    # Initialize object. Inherit methods from object Module
    def __init__(self,tsim,dt,x0,p):
        Module.__init__(self,tsim,dt,x0,p)
        self.D = csg_shape.csg_shape(self.p)
        self.d = csg_fun.outdoorWeather_xls_st2end(self.p)
        if self.p['ctl_vent_type'] == 'timebasedControl' and self.p['ctl_blank_type'] == 'timebasedControl':
            self.U = csg_fun.ctl_csg_pre(self.p)
        else:
            self.U ={}
        self.ext={'ext_Heat_air': 0,
          'ext_co2':0,
          'ext_f':0,
          'ext_vp':0,
          'ext_par':0,
          'ext_nir':0}
        self.u = {'directHeater':0,
                  'extCO2':0,
                  'fan':0,
                  'lamp':0}

        # capacity and mass
        self.C_air = self.p['capacity_air']*1000
        self.m_air = self.p['density_air']*self.D.Vair
        self.C_flo = self.p['capacity_soil'][0]*1000
        self.m_flo = self.p['thickness_soil'][0]*self.D.area_floor*self.p['density_soil'][0]
        self.C_so1 = self.p['capacity_soil'][1] * 1000 *(1-self.p['Water_con'])+self.p['Water_con']*self.p['capacity_water']* 1000
        self.m_so1 = self.p['thickness_soil'][1] * self.D.area_floor * self.p['density_soil'][1]
        self.C_so2 = self.p['capacity_soil'][2] * 1000 *(1-self.p['Water_con'])+self.p['Water_con']*self.p['capacity_water']* 1000
        self.m_so2 = self.p['thickness_soil'][2] * self.D.area_floor * self.p['density_soil'][2]
        self.C_so3 = self.p['capacity_soil'][3] * 1000 *(1-self.p['Water_con'])+self.p['Water_con']*self.p['capacity_water']* 1000
        self.m_so3 = self.p['thickness_soil'][3] * self.D.area_floor * self.p['density_soil'][3]
        self.C_so4 = self.p['capacity_soil'][4] * 1000 *(1-self.p['Water_con'])+self.p['Water_con']*self.p['capacity_water']* 1000
        self.m_so4 = self.p['thickness_soil'][4] * self.D.area_floor * self.p['density_soil'][4]
        self.C_so5 = self.p['capacity_soil'][5] * 1000 *(1-self.p['Water_con'])+self.p['Water_con']*self.p['capacity_water']* 1000
        self.m_so5 = self.p['thickness_soil'][5] * self.D.area_floor * self.p['density_soil'][5]
        self.C_can = self.p['capacity_canopy']*1000

        self.C_wali = self.p['capacity_wall'][0]*1000
        self.m_wali = self.p['thickness_wall'][0]*self.D.area_wall*self.p['density_wall'][0]
        self.C_wal1 = self.p['capacity_wall'][1] * 1000
        self.m_wal1 = self.p['thickness_wall'][1] * self.D.area_wall * self.p['density_wall'][1]
        self.C_wal2 = self.p['capacity_wall'][2] * 1000
        self.m_wal2 = self.p['thickness_wall'][2] * self.D.area_wall * self.p['density_wall'][2]
        self.C_wal3 = self.p['capacity_wall'][3] * 1000
        self.m_wal3 = self.p['thickness_wall'][3] * self.D.area_wall * self.p['density_wall'][3]
        self.C_wal4 = self.p['capacity_wall'][4] * 1000
        self.m_wal4 = self.p['thickness_wall'][4] * self.D.area_wall * self.p['density_wall'][4]
        self.C_wal5 = self.p['capacity_wall'][5] * 1000
        self.m_wal5 = self.p['thickness_wall'][5] * self.D.area_wall * self.p['density_wall'][5]
        self.C_wale = self.p['capacity_wall'][6] * 1000
        self.m_wale = self.p['thickness_wall'][6] * self.D.area_wall * self.p['density_wall'][6]
        self.C_rofi = self.p['capacity_roof'][0]*1000
        self.m_rofi = self.p['thickness_roof'][0]*self.D.area_nroof*self.p['density_roof'][0]
        self.C_rof1 = self.p['capacity_roof'][1] * 1000
        self.m_rof1 = self.p['thickness_roof'][1] * self.D.area_nroof * self.p['density_roof'][1]
        self.C_rofe = self.p['capacity_roof'][2] * 1000
        self.m_rofe = self.p['thickness_roof'][2] * self.D.area_nroof * self.p['density_roof'][2]
        self.C_cov = self.p['capacity_cover']*1000
        self.m_cov = self.p['thickness_cover']*self.D.area_cover*self.p['density_cover']
        self.C_blanki = self.p['capacity_blanket'][0]*1000
        self.m_blanki = self.p['thickness_blanket'][0]*self.D.area_cover*self.p['density_blanket'][0]
        self.C_blanke = self.p['capacity_blanket'][-1] * 1000
        self.m_blanke = self.p['thickness_blanket'][-1] * self.D.area_cover * self.p['density_blanket'][-1]


    # System of algebraic and differential equation(s)
    def diff(self,_t,_y0):
        for vi, vn in enumerate(self.p['StateVariable']):
            setattr(self,vn,_y0[vi])
        # control parameters
        u_blanket, u_vent, u_venttopbot, u_ventside, Vent, H_airvent = csg_fun.ctl_csg1(self.p, self.D, self.d,
                                                                                       self.T_air, _t,self.U)
        # mass of LAI
        self.m_can = self.p['thickness_canopy'] * self.LAI * self.D.area_floor * self.p['density_canopy']
        # outdoor direct and diffuse Radiation
        Rad_dir =self.p['r']*self.d['f_Rad'](_t)*(1-np.exp(-20*self.D.f_alt(_t))) #* (1-r_shadow)
        Rad_dif = self.d['f_Rad'](_t) - self.p['r'] * self.d['f_Rad'](_t) * (1 - np.exp(-20 * self.D.f_alt(_t)))
        # for extra fan vent
        Vent = Vent+ self.ext['ext_f']
        H_airvent = self.D.area_floor * Vent * (self.T_air - self.d['f_Tem'](_t)) * self.p['capacity_air'] * self.p['density_air'] * 1000
        ## radiation absorption
        Sa = 1/(1+np.exp(10**16 * (self.D.angle_nroof-self.D.f_alt(_t))))
        R_glb_air = self.p['R_glb_air']*self.p['GLB_abs_str']*self.d['f_Rad'](_t)
        R_par_flo = self.p['PARabs_soil'][0] * self.D.area_floor * Rad_dir * self.p['r_PAR'] * self.D.f_Tran(_t) * np.exp(-self.p['K_par'] * self.LAI) * (1 - u_blanket) \
                    + self.p['difPARabs_soil'][0] * self.D.area_floor * Rad_dif * self.p['r_PAR'] * self.p['diftran'] * np.exp(-self.p['K_par'] * self.LAI) * (1 - u_blanket)\
                    + self.p['PARabs_soil'][0] * self.D.area_floor * self.ext['ext_par'] * np.exp(-self.p['K_par'] * self.LAI)
        R_nir_flo = self.p['NIRabs_soil'][0] * self.D.area_floor * Rad_dir * (
                    1 - self.p['r_PAR']) * self.D.f_Tran(_t) * np.exp(-self.p['K_nir'] * self.LAI) * (1 - u_blanket) \
                    + self.p['difNIRabs_soil'][0] * self.D.area_floor * Rad_dif * (1 - self.p['r_PAR']) * self.p[
                        'diftran'] * np.exp(-self.p['K_nir'] * self.LAI) * (1 - u_blanket) \
                    + self.p['NIRabs_soil'][0] * self.D.area_floor * self.ext['ext_nir'] * np.exp(-self.p['K_nir'] * self.LAI)
        R_par_can = (Rad_dir * self.D.f_Tran(_t) + Rad_dif * self.p['diftran'])*self.p['r_PAR'] * (1-np.exp(-self.p['K_par'] * self.LAI)) * (1 - u_blanket) * self.D.area_floor + self.ext['ext_par'] *(1-np.exp(-self.p['K_par'] * self.LAI))
        R_nir_can = (Rad_dir * self.D.f_Tran(_t) + Rad_dif * self.p['diftran'])*(1-self.p['r_PAR']) * (1-np.exp(-self.p['K_par'] * self.LAI)) * (1 - u_blanket) * self.D.area_floor + self.ext['ext_nir'] *(1-np.exp(-self.p['K_nir'] * self.LAI))
        Rad_aboveCan_evap = (Rad_dir * self.D.f_Tran(_t) + Rad_dif * self.p['diftran']) * (1 - u_blanket) + self.ext['ext_par'] + self.ext['ext_nir']
        Rad_aboveCan_pho =  R_par_can/self.D.area_floor

        R_par_wali = self.p['PARabs_wall'][0] * max(0,(self.D.area_wall * np.cos(self.D.f_alt(_t)) -self.D.area_nroof*np.sin(self.D.f_alt(_t)-self.D.angle_nroof)* Sa)) \
                    * csg_fun.zerodivided0(np.array([Rad_dir]),np.array([np.sin(self.D.f_alt(_t))])) * self.p['r_PAR'] * self.D.f_Tran(_t)*(1-u_blanket) * np.exp(-self.p['K_par'] * self.LAI)\
                    + self.p['difPARabs_wall'][0] * self.D.area_wall * Rad_dif * self.p['r_PAR'] * self.p['diftran'] * (1-u_blanket)* np.exp(-self.p['K_par'] * self.LAI)
        R_nir_wali = self.p['NIRabs_wall'][0] * max(0,(self.D.area_wall * np.cos(self.D.f_alt(_t)) -self.D.area_nroof*np.sin(self.D.f_alt(_t)-self.D.angle_nroof) * Sa)) \
                     * csg_fun.zerodivided0(np.array([Rad_dir]),np.array([np.sin(self.D.f_alt(_t))])) * (1-self.p['r_PAR']) * self.D.f_Tran(_t)*(1-u_blanket) * np.exp(-self.p['K_nir'] * self.LAI)\
                    + self.p['difNIRabs_wall'][0] * self.D.area_wall * Rad_dif * (1-self.p['r_PAR']) * self.p['diftran'] * (1-u_blanket)* np.exp(-self.p['K_nir'] * self.LAI)
        R_glb_wale = self.p['difPARabs_wall'][-1] * self.D.area_wall * Rad_dif * self.p['r_PAR'] + self.p['difNIRabs_wall'][-1] * self.D.area_wall * Rad_dif * (1-self.p['r_PAR'])
        R_glb_rofi = self.p['PARabs_roof'][0] * self.D.area_nroof * np.sin(self.D.angle_nroof-self.D.f_alt(_t))*(1-Sa) * csg_fun.zerodivided0(np.array([Rad_dir]),np.array([np.sin(self.D.f_alt(_t))]))\
                     * self.p['r_PAR']* self.D.f_Tran(_t)*(1-u_blanket) + \
                      self.p['NIRabs_roof'][0] * self.D.area_nroof * np.sin(self.D.angle_nroof - self.D.f_alt(_t)) * (1 - Sa) * csg_fun.zerodivided0(np.array([Rad_dir]),np.array([np.sin(self.D.f_alt(_t))])) \
                      * (1-self.p['r_PAR']) * self.D.f_Tran(_t) * (1 - u_blanket)\
                     + self.p['difPARabs_roof'][0] * self.D.area_nroof * Rad_dif * self.p['r_PAR']* self.p['diftran'] * (1 - u_blanket) + self.p['difNIRabs_roof'][0]* self.D.area_nroof * Rad_dif * (1-self.p['r_PAR'])* self.p['diftran'] * (1 - u_blanket)
        R_glb_rofe = self.p['PARabs_roof'][-1] * self.D.area_nroof * np.sin(-self.D.angle_nroof+self.D.f_alt(_t))*Sa * csg_fun.zerodivided0(np.array([Rad_dir]),np.array([np.sin(self.D.f_alt(_t))]))\
                     * self.p['r_PAR'] + \
                      self.p['NIRabs_roof'][-1] * self.D.area_nroof * np.sin(-self.D.angle_nroof + self.D.f_alt(_t)) * Sa * csg_fun.zerodivided0(np.array([Rad_dir]), np.array([np.sin(self.D.f_alt(_t))])) \
                      * (1-self.p['r_PAR']) \
                     + self.p['difPARabs_roof'][-1] * self.D.area_nroof * Rad_dif * self.p['r_PAR'] + self.p['difNIRabs_roof'][-1]* self.D.area_nroof * Rad_dif * (1-self.p['r_PAR'])
        R_glb_cov = (1-self.D.f_Tran(_t)) * (self.D.area_floor-self.D.area_nroof*np.cos(self.D.angle_nroof)) * Rad_dir * (1-u_blanket)\
                    + (1-self.p['diftran']) * self.D.area_cover * Rad_dif * (1-u_blanket)
        R_glb_blanke = self.p['PARabs_blanket'][-1] * (self.D.area_floor-self.D.area_nroof*np.cos(self.D.angle_nroof)) * Rad_dir * self.p['r_PAR'] * u_blanket + \
                       self.p['NIRabs_blanket'][-1] * (self.D.area_floor - self.D.area_nroof * np.cos(self.D.angle_nroof)) * Rad_dir * (1-self.p['r_PAR']) * u_blanket +\
                        self.p['difPARabs_blanket'][-1] * self.D.area_cover * Rad_dif * self.p['r_PAR'] * u_blanket +\
                        self.p['NIRabs_blanket'][-1] * self.D.area_cover * Rad_dif * (1-self.p['r_PAR']) * u_blanket

        ## convective heat exchange
        H_cov_air = self.p['cin']*(self.T_cov-self.T_air)*self.D.area_cover
        H_air_cov = -H_cov_air
        H_rofi_air = self.p['cin']*(self.T_rofi-self.T_air)*self.D.area_nroof
        H_air_rofi = -H_rofi_air
        H_wali_air = self.p['cin']*(self.T_wali-self.T_air)*self.D.area_wall
        H_air_wali = -H_wali_air
        H_flo_air = self.p['cin']*(self.T_floor-self.T_air)*self.D.area_floor
        H_air_flo = -H_flo_air
        H_can_air = 2*self.p['c_can_air']*(self.T_can-self.T_air)*self.LAI * self.D.area_floor
        H_air_can = -H_can_air
        H_out_wale = self.p['cout1'] + self.p['cout2'] *(self.d['f_Wind'](_t))**self.p['cout3']*(self.d['f_Tem'](_t)-self.T_wale) * self.D.area_wall
        H_out_rofe = self.p['cout1'] + self.p['cout2'] *(self.d['f_Wind'](_t))**self.p['cout3']*(self.d['f_Tem'](_t)-self.T_rofe) * self.D.area_nroof
        H_out_cov = self.p['cout1'] + self.p['cout2'] *(self.d['f_Wind'](_t))**self.p['cout3']*(self.d['f_Tem'](_t)-self.T_cov) * self.D.area_cover * (1-u_blanket)
        H_out_blanke = self.p['cout1'] + self.p['cout2'] *(self.d['f_Wind'](_t))**self.p['cout3']*(self.d['f_Tem'](_t)-self.T_blanke) * self.D.area_cover * u_blanket
        H_out_blanki = self.p['cout1'] + self.p['cout2'] *(self.d['f_Wind'](_t))**self.p['cout3']*(self.d['f_Tem'](_t)-self.T_blanki) * self.D.area_cover * (1-u_blanket)
        ## ventilation
        V_out_air = -H_airvent

        ## FIR exchange
        R_cov_flo = self.D.AVf_flo_cov * self.p['FIRabs_cover']*self.p['FIRabs_soil'][0] * self.p['Sigma'] * np.exp(-self.p['K_fir']*self.LAI) * ((self.T_cov+273.15)**4-(self.T_floor+273.15)**4)
        R_flo_cov = -R_cov_flo
        R_rofi_flo = self.D.AVf_rofi_flo * self.p['FIRabs_roof'][0]*self.p['FIRabs_soil'][0] * self.p['Sigma'] * np.exp(-self.p['K_fir']*self.LAI) * ((self.T_rofi+273.15)**4-(self.T_floor+273.15)**4)
        R_flo_rofi = -R_rofi_flo
        R_blanki_flo = self.D.AVf_flo_blanki * self.p['FIRabs_blanket'][0]*self.p['FIRabs_soil'][0] * self.p['Sigma'] * np.exp(-self.p['K_fir']*self.LAI) * self.p['FIRtran_cover'] * u_blanket\
                        * ((self.T_blanki+273.15)**4-(self.T_floor+273.15)**4)
        R_flo_blanki = -R_blanki_flo
        R_wali_flo = self.D.AVf_wali_flo * self.p['FIRabs_wall'][0]*self.p['FIRabs_soil'][0] * self.p['Sigma']  * np.exp(-self.p['K_fir']*self.LAI) * ((self.T_wali+273.15)**4-(self.T_floor+273.15)**4)
        R_flo_wali = -R_wali_flo

        R_can_flo = self.D.AVf_flo_can * (1-np.exp(-self.p['K_fir']*self.LAI)) * self.p['FIRabs_soil'][0] * self.p['Sigma'] * (
                    (self.T_can + 273.15) ** 4 - (self.T_floor + 273.15) ** 4)
        R_flo_can = -R_can_flo
        R_sky_flo = self.D.AVf_flo_sky * self.p['FIRabs_soil'][0] * self.p['Sigma'] * np.exp(-self.p['K_fir']*self.LAI) * self.p['FIRtran_cover'] * (1-u_blanket) * ((self.d['f_Tsky'](_t)+273.15)**4-(self.T_floor+273.15)**4)

        R_cov_can = self.D.AVf_cov_can * self.p['FIRabs_cover'] * (1-np.exp(-self.p['K_fir']*self.LAI)) * self.p['Sigma'] * (
                    (self.T_cov + 273.15) ** 4 - (self.T_can + 273.15) ** 4)
        R_can_cov = - R_cov_can

        R_rofi_can = self.D.AVf_rofi_can * self.p['FIRabs_roof'][0] * (1-np.exp(-self.p['K_fir']*self.LAI)) * self.p['Sigma'] * (
                    (self.T_rofi + 273.15) ** 4 - (self.T_can + 273.15) ** 4)
        R_can_rofi = - R_rofi_can

        R_blanki_can = self.D.AVf_blanki_can * self.p['FIRabs_blanket'][0] * (1-np.exp(-self.p['K_fir']*self.LAI)) * self.p[
            'Sigma'] * ((self.T_blanki + 273.15) ** 4 - (self.T_can + 273.15) ** 4) * self.p[
                           'FIRtran_cover'] * u_blanket
        R_can_blanki = - R_blanki_can

        R_wali_can = self.D.AVf_wali_can * self.p['FIRabs_wall'][0] * (1-np.exp(-self.p['K_fir']*self.LAI)) * self.p['Sigma'] * (
                    (self.T_wali + 273.15) ** 4 - (self.T_can + 273.15) ** 4)
        R_can_wali = -R_wali_can

        R_sky_can = self.D.AVf_can_sky * (1-np.exp(-self.p['K_fir']*self.LAI)) * self.p['Sigma'] * (
                    (self.d['f_Tsky'](_t) + 273.15) ** 4 - (self.T_can + 273.15) ** 4) * self.p['FIRtran_cover'] * (
                                1 - u_blanket)
        R_cov_wali = self.D.AVf_wali_cov * self.p['FIRabs_cover'] * self.p['FIRabs_wall'][0] * self.p['Sigma'] * ((self.T_cov+273.15)**4-(self.T_wali+273.15)**4) * np.exp(-self.p['K_fir']*self.LAI)
        R_wali_cov = -R_cov_wali
        R_rofi_wali = self.D.AVf_rofi_wali * self.p['FIRabs_roof'][0] * self.p['FIRabs_wall'][0] * self.p['Sigma'] * ((self.T_rofi+273.15)**4-(self.T_wali+273.15)**4)
        R_wali_rofi = -R_rofi_wali
        R_blanki_wali = self.D.AVf_wali_blanki * self.p['FIRabs_blanket'][0] * self.p['FIRabs_wall'][0] * self.p['Sigma'] * ((self.T_blanki+273.15)**4-(self.T_wali+273.15)**4) * np.exp(-self.p['K_fir']*self.LAI) * self.p['FIRtran_cover']*u_blanket
        R_wali_blanki = -R_blanki_wali
        R_sky_wali = self.D.AVf_wali_sky * self.p['FIRabs_wall'][0] * self.p['Sigma'] * ((self.d['f_Tsky'](_t)+273.15)**4-(self.T_wali+273.15)**4) * np.exp(-self.p['K_fir']*self.LAI) * self.p['FIRtran_cover']* (1-u_blanket)
        R_sky_wale = self.D.AVf_wale_sky * self.p['FIRabs_wall'][-1] * self.p['Sigma'] * ((self.d['f_Tsky'](_t)+273.15)**4-(self.T_wale+273.15)**4)
        R_cov_rofi = self.D.AVf_rofi_cov * self.p['FIRabs_cover'] * self.p['FIRabs_roof'][0] * self.p['Sigma'] * ((self.T_cov+273.15)**4-(self.T_rofi+273.15)**4)
        R_rofi_cov = -R_cov_rofi
        R_blanki_rofi = self.D.AVf_rofi_blanki * self.p['FIRabs_blanket'][0] * self.p['FIRabs_roof'][0] * self.p['Sigma'] * ((self.T_blanki+273.15)**4-(self.T_rofi+273.15)**4)*u_blanket
        R_rofi_blanki = -R_blanki_rofi
        R_sky_rofi = self.D.AVf_rofi_sky * self.p['FIRabs_roof'][0] * self.p['Sigma'] * ((self.d['f_Tsky'](_t)+273.15)**4-(self.T_rofi+273.15)**4) * (1-u_blanket)
        R_sky_rofe = self.D.AVf_rofe_sky * self.p['FIRabs_roof'][-1] * self.p['Sigma'] * ((self.d['f_Tsky'](_t)+273.15)**4-(self.T_rofe+273.15)**4)
        R_sky_cov = self.D.AVf_cov_sky * self.p['FIRabs_cover'] * self.p['Sigma'] * ((self.d['f_Tsky'](_t)+273.15)**4-(self.T_cov+273.15)**4)* (1-u_blanket)
        R_sky_blanke = self.D.AVf_blanke_sky * self.p['FIRabs_blanket'][-1] * self.p['Sigma'] * ((self.d['f_Tsky'](_t)+273.15)**4-(self.T_blanke+273.15)**4)* u_blanket
        R_sky_blanki = 1 * self.p['FIRabs_blanket'][1] * self.p['Sigma'] * ((self.d['f_Tsky'](_t)+273.15)**4-(self.T_blanki+273.15)**4)* (1-u_blanket)

        ## conductive heat exchange
        H_so1_flo = csg_fun.conductive(self.p['thickness_soil'][1],self.p['heatcond_soil'][1],self.T_soil1,self.p['thickness_soil'][0],self.p['heatcond_soil'][0],self.T_floor,self.D.area_floor)
        H_flo_so1 = -H_so1_flo
        H_so2_so1 = csg_fun.conductive(self.p['thickness_soil'][2],self.p['heatcond_soil'][2],self.T_soil2,self.p['thickness_soil'][1],self.p['heatcond_soil'][1],self.T_soil1,self.D.area_floor)
        H_so1_so2 = -H_so2_so1
        H_so3_so2 = csg_fun.conductive(self.p['thickness_soil'][3],self.p['heatcond_soil'][3],self.T_soil3,self.p['thickness_soil'][2],self.p['heatcond_soil'][2],self.T_soil2,self.D.area_floor)
        H_so2_so3 = -H_so3_so2
        H_so4_so3 = csg_fun.conductive(self.p['thickness_soil'][4],self.p['heatcond_soil'][4],self.T_soil4,self.p['thickness_soil'][3],self.p['heatcond_soil'][3],self.T_soil3,self.D.area_floor)
        H_so3_so4 = -H_so4_so3
        H_so5_so4 = csg_fun.conductive(self.p['thickness_soil'][5],self.p['heatcond_soil'][5],self.T_soil5,self.p['thickness_soil'][4],self.p['heatcond_soil'][4],self.T_soil4,self.D.area_floor)
        H_so4_so5 = -H_so5_so4
        H_soe_so5 = csg_fun.conductive(0,self.p['heatcond_soil'][5],self.p['T_soilbound'],self.p['thickness_soil'][5],self.p['heatcond_soil'][5],self.T_soil5,self.D.area_floor)
        H_wal1_wali = csg_fun.conductive(self.p['thickness_wall'][1],self.p['heatcond_wall'][1],self.T_wal1,self.p['thickness_wall'][0],self.p['heatcond_wall'][0],self.T_wali,self.D.area_wall)
        H_wali_wal1 = -H_wal1_wali
        H_wal2_wal1 = csg_fun.conductive(self.p['thickness_wall'][2],self.p['heatcond_wall'][2],self.T_wal2,self.p['thickness_wall'][1],self.p['heatcond_wall'][1],self.T_wal1,self.D.area_wall)
        H_wal1_wal2 = -H_wal2_wal1
        H_wal3_wal2 = csg_fun.conductive(self.p['thickness_wall'][3],self.p['heatcond_wall'][3],self.T_wal3,self.p['thickness_wall'][2],self.p['heatcond_wall'][2],self.T_wal2,self.D.area_wall)
        H_wal2_wal3 = -H_wal3_wal2
        H_wal4_wal3 = csg_fun.conductive(self.p['thickness_wall'][4],self.p['heatcond_wall'][4],self.T_wal4,self.p['thickness_wall'][3],self.p['heatcond_wall'][3],self.T_wal3,self.D.area_wall)
        H_wal3_wal4 = -H_wal4_wal3
        H_wal5_wal4 = csg_fun.conductive(self.p['thickness_wall'][5],self.p['heatcond_wall'][5],self.T_wal5,self.p['thickness_wall'][4],self.p['heatcond_wall'][4],self.T_wal4,self.D.area_wall)
        H_wal4_wal5 = -H_wal5_wal4
        H_wale_wal5 = csg_fun.conductive(self.p['thickness_wall'][-1],self.p['heatcond_wall'][-1],self.T_wale,self.p['thickness_wall'][5],self.p['heatcond_wall'][5],self.T_wal5,self.D.area_wall)
        H_wal5_wale = -H_wale_wal5
        H_rof1_rofi = csg_fun.conductive(self.p['thickness_roof'][1],self.p['heatcond_roof'][1],self.T_rof1,self.p['thickness_roof'][0],self.p['heatcond_roof'][0],self.T_rofi,self.D.area_nroof)
        H_rofi_rof1 = -H_rof1_rofi
        H_rofe_rof1 = csg_fun.conductive(self.p['thickness_roof'][-1],self.p['heatcond_roof'][-1],self.T_rofe,self.p['thickness_roof'][1],self.p['heatcond_roof'][1],self.T_rof1,self.D.area_nroof)
        H_rof1_rofe = -H_rofe_rof1
        H_blanki_cov = csg_fun.conductive(self.p['thickness_blanket'][0],self.p['heatcond_blanket'][0],self.T_blanki,self.p['thickness_cover'],self.p['heatcond_cover'],self.T_cov,self.D.area_cover)*u_blanket
        H_cov_blanki = -H_blanki_cov
        H_blanke_blanki = csg_fun.conductive(self.p['thickness_blanket'][-1],self.p['heatcond_blanket'][-1],self.T_blanke,self.p['thickness_blanket'][0],self.p['heatcond_blanket'][0],self.T_blanki,self.D.area_cover)
        H_blanki_blanke = -H_blanke_blanki
        ## latent heat
        dvp, M, L_air_cov, L_air_can, L_air_so2 = csg_fun.vapour(self.p,self.D,self.T_air,self.VP,self.T_cov,self.T_can,self.T_soil2,self.d['f_Tem'](_t),self.d['f_RH'](_t),self.D.area_cover,self.D.area_floor,Rad_aboveCan_evap,self.CO2,self.C_air*self.m_air,self.LAI,Vent,self.ext['ext_vp'])

        # plant model
        MCairbuf = BVtomato_fun.BramVanthoorPhotoSynthesis(self.T_air,Rad_aboveCan_pho,self.CO2,self.LAI,self.C_buf,self.p)
        MCbuffruit_tot, MCbufleaf, MCbuf_stemroot = BVtomato_fun.BramVanthoorPartioning(self.LAI,self.T_air,self.C_buf,self.T_can24,self.T_sum,self.p)
        MCbufair_g = BVtomato_fun.BramVanthoorGrowthRespiration(MCbuffruit_tot,MCbufleaf,MCbuf_stemroot,self.LAI,self.C_buf,self.p)
        MCbuffruit_tot, MCbufleaf, MCbuf_stemroot, MCbufair_g = BVtomato_fun.BoSmoth1(self.C_buf, MCbuffruit_tot, MCbufleaf,MCbuf_stemroot, MCbufair_g)
        Cbufdot = BVtomato_fun.BramVanthoorCarboBuffer(MCairbuf, MCbuffruit_tot, MCbufleaf, MCbuf_stemroot, MCbufair_g)
        MCstemroot_air_m, MCleafair_m, MCfruitair_m, MCorgair_m = BVtomato_fun.BramVanthoorMaintenanceRespiration(MCbufleaf,self.T_air,self.C_stemroot,self.C_leaf,np.array([self.C_fruit1,self.C_fruit2,self.C_fruit3,self.C_fruit4,self.C_fruit5]),self.p)
        MCstemroot_air_m, MCleafair_m, MCfruitair_m, MCorgair_m = BVtomato_fun.BoSmoth2(MCstemroot_air_m, MCleafair_m,
                                                                             MCfruitair_m, self.C_stemroot,self.C_leaf,
                                                                             np.array([self.C_fruit1,self.C_fruit2,self.C_fruit3,self.C_fruit4,self.C_fruit5]), MCbuffruit_tot, MCbufleaf, MCbuf_stemroot)
        MCleafhar = BVtomato_fun.BramVanthoorLeafHarvest(self.C_leaf,self.p)
        Chardot, LAIdot, Cstemrootdot, Cleafdot, Cfruitdot, Nfruitdot = BVtomato_fun.BramVanthoorPlantBuffer(MCbuffruit_tot,MCbufleaf,MCbuf_stemroot,MCstemroot_air_m, MCleafair_m,MCfruitair_m,MCleafhar,self.T_air,self.T_sum,self.T_can24,\
                                                                                                             np.array([self.C_fruit1,self.C_fruit2,self.C_fruit3,self.C_fruit4,self.C_fruit5]),np.array([self.N_fruit1,self.N_fruit2,self.N_fruit3,self.N_fruit4,self.N_fruit5]),self.p)

        # Differential equations
        dTair_dt = (1/self.C_air/self.m_air)*(R_glb_air+H_cov_air+H_rofi_air+H_wali_air+H_flo_air+H_can_air+V_out_air + self.ext['ext_Heat_air'])

        self.p['soil_res'] =(0.64*(Rad_dif+Rad_dir)+57)*csg_fun.switch01(np.array([self.CO2-900]),1)
        dCO2_dt = Vent * self.D.area_floor / self.D.Vair * (self.d['f_CO2'](_t) - self.CO2) + (MCorgair_m - MCairbuf) / \
                  self.p['MCH2O'] * self.p['MCO2'] * self.D.area_floor / self.D.Vair / self.p['eta_ppm_mgm3'] + (
                              self.p['soil_res'] ) * 1000000 / 10000 / 86400 * self.D.area_floor / \
                  self.p['MC'] * self.p['MCO2'] / self.D.Vair / self.p['eta_ppm_mgm3']  \
                  + self.ext['ext_co2'] / self.D.Vair / self.p['eta_ppm_mgm3']
        dLAI_dt = LAIdot
        dTflo_dt = (1/self.C_flo/self.m_flo)*( R_par_flo + R_nir_flo + R_cov_flo + R_rofi_flo + R_blanki_flo + R_wali_flo + R_can_flo + R_sky_flo + H_air_flo + H_so1_flo )
        dTso1_dt = (1/self.C_so1/self.m_so1)*( H_flo_so1 + H_so2_so1 )
        dTso2_dt = (1/self.C_so2/self.m_so2)*( H_so1_so2 + H_so3_so2  + L_air_so2)
        dTso3_dt = (1/self.C_so3/self.m_so3)*( H_so2_so3 + H_so4_so3 )
        dTso4_dt = (1/self.C_so4/self.m_so4)*( H_so3_so4 + H_so5_so4 )
        dTso5_dt = (1/self.C_so5/self.m_so5)*( H_so4_so5 + H_soe_so5 )
        dTcan_dt = (1/self.C_can/self.m_can)*( R_par_can + R_nir_can + R_cov_can + R_rofi_can + R_blanki_can + R_wali_can + R_flo_can + R_sky_can + H_air_can + L_air_can )
        dTwali_dt = (1/self.C_wali/self.m_wali)*( R_par_wali + R_nir_wali + R_cov_wali + R_rofi_wali + R_blanki_wali + R_flo_wali + R_can_wali + R_sky_wali + H_air_wali + H_wal1_wali )
        dTwal1_dt = (1/self.C_wal1/self.m_wal1)*( H_wali_wal1 + H_wal2_wal1 )
        dTwal2_dt = (1/self.C_wal2/self.m_wal2)*( H_wal1_wal2 + H_wal3_wal2 )
        dTwal3_dt = (1/self.C_wal3/self.m_wal3)*( H_wal2_wal3 + H_wal4_wal3 )
        dTwal4_dt = (1/self.C_wal4/self.m_wal4)*( H_wal3_wal4 + H_wal5_wal4 )
        dTwal5_dt = (1/self.C_wal5/self.m_wal5)*( H_wal4_wal5 + H_wale_wal5 )
        dTwale_dt = (1/self.C_wale/self.m_wale)*( R_glb_wale + R_sky_wale + H_out_wale + H_wal5_wale )
        dTrofi_dt = (1/self.C_rofi/self.m_rofi)*( R_glb_rofi + R_cov_rofi + R_blanki_rofi + R_wali_rofi + R_flo_rofi + R_can_rofi + R_sky_rofi + H_air_rofi + H_rof1_rofi )
        dTrof1_dt = (1/self.C_rof1/self.m_rof1)*( H_rofi_rof1 + H_rofe_rof1)
        dTrofe_dt = (1/self.C_rofe/self.m_rofe)*( R_glb_rofe + R_sky_rofe +H_out_rofe + H_rof1_rofe )
        dTcov_dt = (1/self.C_cov/self.m_cov)*( R_glb_cov + R_rofi_cov + H_blanki_cov + R_wali_cov + R_flo_cov + R_can_cov + R_sky_cov +H_air_cov + H_out_cov + L_air_cov )
        dTblanki_dt = (1/self.C_blanki/self.m_blanki)*( H_cov_blanki + R_rofi_blanki + H_blanke_blanki + R_wali_blanki + R_flo_blanki + R_can_blanki + H_out_blanki + R_sky_blanki)
        dTblanke_dt = (1/self.C_blanke/self.m_blanke)*( R_glb_blanke + H_blanki_blanke + R_sky_blanke + H_out_blanke )
        dTsum_dt = self.T_air/86400
        dT24_dt = (self.T_air-self.T_can24)/86400
        dCbuf_dt = Cbufdot
        dCstemroot_dt = Cstemrootdot
        dCleaf_dt = Cleafdot
        dChar_dt = Chardot
        dCfruit1_dt = Cfruitdot[0]
        dCfruit2_dt = Cfruitdot[1]
        dCfruit3_dt = Cfruitdot[2]
        dCfruit4_dt = Cfruitdot[3]
        dCfruit5_dt = Cfruitdot[4]
        dNfruit1_dt = Nfruitdot[0]
        dNfruit2_dt = Nfruitdot[1]
        dNfruit3_dt = Nfruitdot[2]
        dNfruit4_dt = Nfruitdot[3]
        dNfruit5_dt = Nfruitdot[4]
        return np.nan_to_num(np.array([dTair_dt,dvp,dCO2_dt,dLAI_dt,dTflo_dt,dTso1_dt,dTso2_dt,dTso3_dt,
                         dTso4_dt,dTso5_dt,dTcan_dt,dTwali_dt,dTwal1_dt,dTwal2_dt,dTwal3_dt,
                         dTwal4_dt,dTwal5_dt,dTwale_dt,dTrofi_dt,dTrof1_dt,dTrofe_dt,dTcov_dt,
                         dTblanki_dt,dTblanke_dt,dTsum_dt,dT24_dt,dCbuf_dt,dCstemroot_dt,dCleaf_dt,
                         dChar_dt,dCfruit1_dt,dCfruit2_dt,dCfruit3_dt,dCfruit4_dt,dCfruit5_dt,dNfruit1_dt,
                         dNfruit2_dt,dNfruit3_dt,dNfruit4_dt,dNfruit5_dt]),nan=0,posinf=0,neginf=0)
    # Define method 'model'
    def model(self,tspan,dt,x0,p,d,u):
        # Initial conditions
        y0 = self.p['InitialValues']
        # Integration output
        y_int = fcn_euler_forward(self.diff,tspan,y0,h=dt)#,kwargs={'d':1})
        return {'t':y_int['t'], 'T_air':y_int['y'][0,:],'VP': y_int['y'][1,:], 'CO2': y_int['y'][2,:], 'LAI': y_int['y'][3,:], \
              'T_floor': y_int['y'][4,:], 'T_soil1': y_int['y'][5,:], 'T_soil2': y_int['y'][6,:], 'T_soil3': y_int['y'][7,:], 'T_soil4': y_int['y'][8,:], 'T_soil5': y_int['y'][9,:], \
              'T_can': y_int['y'][10,:], \
              'T_wali': y_int['y'][11,:], 'T_wal1': y_int['y'][12,:], 'T_wal2': y_int['y'][13,:], 'T_wal3': y_int['y'][14,:], 'T_wal4': y_int['y'][15,:], 'T_wal5': y_int['y'][16,:], 'T_wale': y_int['y'][17,:], \
              'T_rofi': y_int['y'][18,:], 'T_rof1': y_int['y'][19,:], 'T_rofe': y_int['y'][20,:], \
              'T_cov': y_int['y'][21,:], \
              'T_blanki': y_int['y'][22,:], 'T_blanke': y_int['y'][23,:], \
              'T_sum': y_int['y'][24,:], 'T_can24': y_int['y'][25,:], \
              'C_buf': y_int['y'][26,:], 'C_stemroot': y_int['y'][27,:], 'C_leaf': y_int['y'][28,:], 'C_har': y_int['y'][29,:], \
              'C_fruit1': y_int['y'][30,:], 'C_fruit2': y_int['y'][31,:], 'C_fruit3': y_int['y'][32,:], 'C_fruit4': y_int['y'][33,:], 'C_fruit5': y_int['y'][34,:], \
              'N_fruit1': y_int['y'][35,:], 'N_fruit2': y_int['y'][36,:], 'N_fruit3': y_int['y'][37,:], 'N_fruit4': y_int['y'][38,:], 'N_fruit5': y_int['y'][39,:]}


    def resetModel(self):
        self.D = csg_shape.csg_shape(self.p)
        self.d = csg_fun.outdoorWeather_xls_st2end(self.p)
        self.C_air = self.p['capacity_air'] * 1000
        self.m_air = self.p['density_air'] * self.D.Vair
        self.C_flo = self.p['capacity_soil'][0] * 1000
        self.m_flo = self.p['thickness_soil'][0] * self.D.area_floor * self.p['density_soil'][0]
        self.C_so1 = self.p['capacity_soil'][1] * 1000 * (1 - self.p['Water_con']) + self.p['Water_con'] * self.p[
            'capacity_water'] * 1000
        self.m_so1 = self.p['thickness_soil'][1] * self.D.area_floor * self.p['density_soil'][1]
        self.C_so2 = self.p['capacity_soil'][2] * 1000 * (1 - self.p['Water_con']) + self.p['Water_con'] * self.p[
            'capacity_water'] * 1000
        self.m_so2 = self.p['thickness_soil'][2] * self.D.area_floor * self.p['density_soil'][2]
        self.C_so3 = self.p['capacity_soil'][3] * 1000 * (1 - self.p['Water_con']) + self.p['Water_con'] * self.p[
            'capacity_water'] * 1000
        self.m_so3 = self.p['thickness_soil'][3] * self.D.area_floor * self.p['density_soil'][3]
        self.C_so4 = self.p['capacity_soil'][4] * 1000 * (1 - self.p['Water_con']) + self.p['Water_con'] * self.p[
            'capacity_water'] * 1000
        self.m_so4 = self.p['thickness_soil'][4] * self.D.area_floor * self.p['density_soil'][4]
        self.C_so5 = self.p['capacity_soil'][5] * 1000 * (1 - self.p['Water_con']) + self.p['Water_con'] * self.p[
            'capacity_water'] * 1000
        self.m_so5 = self.p['thickness_soil'][5] * self.D.area_floor * self.p['density_soil'][5]
        self.C_can = self.p['capacity_canopy'] * 1000
        self.C_wali = self.p['capacity_wall'][0] * 1000
        self.m_wali = self.p['thickness_wall'][0] * self.D.area_wall * self.p['density_wall'][0]
        self.C_wal1 = self.p['capacity_wall'][1] * 1000
        self.m_wal1 = self.p['thickness_wall'][1] * self.D.area_wall * self.p['density_wall'][1]
        self.C_wal2 = self.p['capacity_wall'][2] * 1000
        self.m_wal2 = self.p['thickness_wall'][2] * self.D.area_wall * self.p['density_wall'][2]
        self.C_wal3 = self.p['capacity_wall'][3] * 1000
        self.m_wal3 = self.p['thickness_wall'][3] * self.D.area_wall * self.p['density_wall'][3]
        self.C_wal4 = self.p['capacity_wall'][4] * 1000
        self.m_wal4 = self.p['thickness_wall'][4] * self.D.area_wall * self.p['density_wall'][4]
        self.C_wal5 = self.p['capacity_wall'][5] * 1000
        self.m_wal5 = self.p['thickness_wall'][5] * self.D.area_wall * self.p['density_wall'][5]
        self.C_wale = self.p['capacity_wall'][6] * 1000
        self.m_wale = self.p['thickness_wall'][6] * self.D.area_wall * self.p['density_wall'][6]
        self.C_rofi = self.p['capacity_roof'][0] * 1000
        self.m_rofi = self.p['thickness_roof'][0] * self.D.area_nroof * self.p['density_roof'][0]
        self.C_rof1 = self.p['capacity_roof'][1] * 1000
        self.m_rof1 = self.p['thickness_roof'][1] * self.D.area_nroof * self.p['density_roof'][1]
        self.C_rofe = self.p['capacity_roof'][2] * 1000
        self.m_rofe = self.p['thickness_roof'][2] * self.D.area_nroof * self.p['density_roof'][2]
        self.C_cov = self.p['capacity_cover'] * 1000
        self.m_cov = self.p['thickness_cover'] * self.D.area_cover * self.p['density_cover']
        self.C_blanki = self.p['capacity_blanket'][0] * 1000
        self.m_blanki = self.p['thickness_blanket'][0] * self.D.area_cover * self.p['density_blanket'][0]
        self.C_blanke = self.p['capacity_blanket'][-1] * 1000
        self.m_blanke = self.p['thickness_blanket'][-1] * self.D.area_cover * self.p['density_blanket'][-1]

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    time_start = time.time()
    p = example.parameters()
    dt = p['dt']
    tsim = np.arange(0,time.mktime(time.strptime(p['EndTime'],"%Y-%m-%dT%H:%M"))-time.mktime(time.strptime(p['StartTime'],"%Y-%m-%dT%H:%M")),p['dtsim'])
    x0={}
    for i,stateN in enumerate(p['StateVariable']):
        x0[stateN]=p['InitialValues'][i]
    lg = CSG_Climate(tsim,dt,x0,p)
    # Run model
    tspan = (tsim[0],tsim[-1])
    y = lg.run(tspan)
    plt.plot(y['t'],y['T_air'])
    plt.show()
