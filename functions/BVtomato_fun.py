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
@brief Bram Vanthoor Photosynthesis Model- Vanthoor, B. (2011). A model-based greenhouse design method. Wageningen PhD thesis.
@authors: Bo Zhou（周波）

"""
import numpy as np
import math

def BramVanthoorPhotoSynthesis(Tcan, R_PARcan, CO2air, LAI, Cbuf, p):
    Cbuf_max = min(20e3, 20e3 * LAI / p['LAI_Max'])
    Jmax25 = 210 * LAI
    Pg_L, VC2, FRESP, J = Photosynthesis3(Tcan,R_PARcan,CO2air,Jmax25,p['THETA'],LAI)
    MCairbuf_pot = 30e-3 * Pg_L
    h_Buf_MCAirBuf = SmoothIfElse(Cbuf, Cbuf_max, 5e-4)
    MCAirBuf = h_Buf_MCAirBuf * MCairbuf_pot
    return MCAirBuf
def Photosynthesis3(Tcan, PARABS, CO2air, JMAX25, THETA, LAI):
    VCMAX25 = 0.5 * JMAX25
    RD25 = 1.25
    KC25 = 310
    KO25 = 155
    p_O2can = 210
    LGHTCON = 4.6
    R = 8.314
    F = 0.23
    S = 710
    H = 220000
    eta_CO2air_CO2can = 0.67
    EC = 59356
    EO = 35948
    EJ = 37000
    EVC = 58520
    MCO2 = 44e-3
    CO2can = eta_CO2air_CO2can * CO2air
    Tcan = Tcan + 273.15
    T25 = 25 + 273.15
    X = (Tcan - T25) / (Tcan * R * T25)

    KC = KC25 * np.exp(EC * X)
    KO = KO25 * np.exp(EO * X)
    VCMAX = VCMAX25 * np.exp(EVC * X)
    b = 1.7 * 20 * (1 - 1 / LAI)
    GAMMA1 = 1.7 / LAI * (Tcan - 273.15) + b
    if GAMMA1 < 5:
        GAMMA1 = 5
    D1 = 1 + np.exp((S - H / T25) / R)
    D2 = 1 + np.exp((S - H / Tcan) / R)
    TJMAX = np.exp(EJ * X) * D1 / D2
    JMAX = JMAX25 * TJMAX
    alfa = 0.385
    EFFRAD = PARABS * LGHTCON * alfa
    J = (JMAX + EFFRAD - math.sqrt(abs((JMAX + EFFRAD) ** 2 - 4 * THETA * EFFRAD * JMAX))) / (2 * THETA)
    JC = J
    VC2 = JC / 4 * (CO2can - GAMMA1) / (CO2can + 2 * GAMMA1)  # Used by Intkam
    VC = VC2
    FRESP = VC * GAMMA1 / CO2can
    FARPHG = VC - FRESP
    Pg_L = FARPHG
    return Pg_L, VC2, FRESP, J

def SmoothIfElse(Value, SwitchValue, Slope):
    SmoothValue = 1 / (1 + np.exp(Slope * (Value - SwitchValue)))
    return SmoothValue
def BoSmoth1(Cbuf,MCbuffruit_tot,MCbufleaf,MCbuf_stemroot,MCbufair_g):
    h_Buf_empty = SmoothIfElse(Cbuf, (MCbuffruit_tot + MCbufleaf + MCbuf_stemroot + MCbufair_g), -20e10)
    MCbuffruit_tot = h_Buf_empty * MCbuffruit_tot
    MCbufleaf = h_Buf_empty * MCbufleaf
    MCbuf_stemroot = h_Buf_empty * MCbuf_stemroot
    MCbufair_g = h_Buf_empty * MCbufair_g
    return MCbuffruit_tot,MCbufleaf,MCbuf_stemroot,MCbufair_g
def BoSmoth2(MCstemroot_air_m, MCleafair_m, MCfruitair_m,Cstemroot,Cleaf,Cfruit,MCbuffruit_tot,MCbufleaf,MCbuf_stemroot):
    h_stemroot_empty = SmoothIfElse(Cstemroot + MCbuf_stemroot, MCstemroot_air_m, -20e10)
    MCstemroot_air_m = h_stemroot_empty * MCstemroot_air_m
    h_leaf_empty = SmoothIfElse(Cleaf + MCbufleaf, MCleafair_m, -20e10)
    MCleafair_m = h_leaf_empty * MCleafair_m
    h_fruit_empty = SmoothIfElse(sum(Cfruit) + MCbuffruit_tot, sum(MCfruitair_m), -20e10)
    MCfruitair_m = h_fruit_empty * MCfruitair_m
    MCorgair_m = sum(MCfruitair_m) + MCleafair_m + MCstemroot_air_m
    return MCstemroot_air_m, MCleafair_m, MCfruitair_m, MCorgair_m
def BramVanthoorPartioning(LAI, Tcan, Cbuf, Tcan24, Tsum,p):
    Cbuf_max = min(20e3, 20e3 * LAI / p['LAI_Max'])
    h_Buf_MCBufOrg = SmoothIfElse(Cbuf, 0.05 * Cbuf_max, -20e-3)
    f1 = 1 / p['Tsum_needed'] * Tsum
    f2 = 0
    f3 = 1 / p['Tsum_needed'] * (Tsum - p['Tsum_needed'])
    a1_s = 0.5*(f2 + f1 + math.sqrt((abs(f2 - f1))**2 +1e-4))
    a2_s = 0.5*(f2 + f3 + math.sqrt((abs(f2 - f3))**2 +1e-4))
    g_MCBufFruit_Tsum = a1_s - a2_s
    h_Tcan_Char, h_Tcan24_Char = GrowthInhibition2(Tcan, Tcan24,p)
    g_MCBufOrg_Tcan24 = 0.047 * Tcan24 + 0.060
    MCbuffruit_tot =   h_Buf_MCBufOrg * h_Tcan24_Char * h_Tcan_Char * g_MCBufOrg_Tcan24 * g_MCBufFruit_Tsum * p['rg_fruit']
    MCbufleaf =        h_Buf_MCBufOrg * h_Tcan24_Char *               g_MCBufOrg_Tcan24 *                     p['rg_leaf']
    MCbuf_stemroot =   h_Buf_MCBufOrg * h_Tcan24_Char *               g_MCBufOrg_Tcan24 *                     p['rg_stemroot']
    return MCbuffruit_tot,MCbufleaf,MCbuf_stemroot

def GrowthInhibition2(Tcan,Tcan24,p):
    [h_Tcan_24, h_Tcan_24_der] = inhibition_continue(Tcan24, p['T24_S1'], p['T24_b1'], p['T24_S2'], p['T24_b2'])
    [h_Tcan_photo, h_Tcan_photo_der] = inhibition_continue(Tcan24, p['T24_S1'], p['T24_b1'], p['T24_S2'], p['T24_b2'])
    [h_Tcan_inst, h_Tcan_inst_der] = inhibition_continue(Tcan, p['Tinst_S1'], p['Tinst_b1'], p['Tinst_S2'], p['Tinst_b2'])
    MCaircrp_mean = 0
    h_DIF = 0
    return h_Tcan_inst,h_Tcan_24

def inhibition_continue(X, S1,b1,S2,b2):
    h_c1 = 1 / (1 + np.exp(-S1 * (X - b1)))
    h_c2 = 1 / (1 + np.exp(-S2 * (X - b2)))
    h_c = h_c1 * h_c2
    h_c_der = (1 + np.exp(-S1 * (X - b1)))**-2 * S1 * np.exp(-S1 * (X - b1)) * 1 / (1 + np.exp(-S2 * (X - b2)))\
    + 1 / (1 + np.exp(-S1 * (X - b1))) * (1 + np.exp(-S2 * (X - b2)))**-2 * S2 * np.exp(-S2 * (X - b2))
    return h_c,h_c_der

def BramVanthoorGrowthRespiration(MCbuffruit_tot,MCbufleaf,MCbuf_stemroot,LAI,Cbuf,p):
    Cbuf_max = min(20e3, 20e3 * LAI / p['LAI_Max'])
    h_Buf_MCBufOrg = SmoothIfElse(Cbuf, 0.05 * Cbuf_max, -20e-3)
    MCfruitair_g = p['cfruit_g'] * MCbuffruit_tot
    MCleafair_g = p['cleaf_g'] * MCbufleaf
    MCstemroot_air_g = p['cstemroot_g'] * MCbuf_stemroot
    MCbufair_g = MCfruitair_g + MCleafair_g + MCstemroot_air_g
    MCbufair_g = h_Buf_MCBufOrg * MCbufair_g
    return MCbufair_g

def BramVanthoorCarboBuffer(MCAirBuf, MCbuffruit_tot, MCbufleaf, MCbuf_stemroot, MCbufair_g):
    Cbufdot = MCAirBuf - MCbuffruit_tot - MCbufleaf - MCbuf_stemroot - MCbufair_g
    return Cbufdot

def BramVanthoorMaintenanceRespiration(MCbufleaf,Tcan,Cstemroot,Cleaf,Cfruit,p):
    MCfruitair_m = np.zeros(p['NrBox'])
    for i in np.arange(0,p['NrBox']):
        MCfruitair_m_without = Respiration(Cfruit[i], Tcan, 0, p['cfruit_m'], p['Q10m'], 0)
        MCfruitair_m[i] = p['Resp_fac'] * MCfruitair_m_without
    MCfruitair_m_tot = sum(MCfruitair_m)
    MCleafair_m = min(p['Resp_fac'] * Respiration(Cleaf, Tcan, 0, p['cleaf_m'], p['Q10m'], 0), MCbufleaf)
    MCstemroot_air_m = p['Resp_fac'] * Respiration(Cstemroot, Tcan, 0, p['cstemroot_m'], p['Q10m'], 0)
    MCorgair_m = MCfruitair_m_tot + MCleafair_m + MCstemroot_air_m
    return MCstemroot_air_m, MCleafair_m, MCfruitair_m,MCorgair_m

def Respiration(Ccrp, Tcan, MCaircrp, rm25, Q10m, c_g):
    Tref = 25
    MCcrpair_m = rm25 * Q10m**(0.1 * (Tcan - Tref)) * Ccrp
    MCcrpair_m_der = 0
    Slope = 10e7
    ValueIfElse = SmoothIfElse(MCaircrp, MCcrpair_m, Slope)
    MCcrpair_g = c_g * (MCaircrp - MCcrpair_m) * (ValueIfElse)
    return MCcrpair_m

def BramVanthoorLeafHarvest(Cleaf,p):
    Cleaf_Max = p['LAI_Max'] / p['SLA']
    MCleafhar = 0.001 * SmoothIfElse(Cleaf, Cleaf_Max, -5e-5) * (Cleaf - Cleaf_Max)
    MCleafhar = max(MCleafhar, 0)
    return MCleafhar

def BramVanthoorPlantBuffer(MCbuffruit_tot,MCbufleaf,MCbuf_stemroot,MCstemroot_air_m, MCleafair_m, MCfruitair_m,MCleafhar,Tcan,Tsum,Tcan24,Cfruit,Nfruit,p):
    r_dev = (-0.066 + 0.1 * Tcan24) / (100 * 86400)
    FGP = (1 / (r_dev * 86400))
    t_Kon = np.zeros(p['NrBox'])
    for i in np.arange(0,p['NrBox']):
        t_Kon[i] = (2 * i  + 1) / (2 * p['NrBox']) * FGP
    h_Tcansum_Cfruit = np.ones(p['NrBox'])
    M =  -4.93 + 0.548 * FGP
    C =10
    B = 1 / (2.44 + 0.403 * M)
    GR = C * np.exp(-np.exp(-B * (t_Kon - M))) * B * np.exp(-B * (t_Kon - M))
    Wfruit_pot = C * np.exp(-np.exp(-B * (t_Kon - M))) * 1000
    MNfruit = []
    for n in Nfruit:
        MNfruit.append(r_dev * n * p['NrBox'] * SmoothIfElse(Tsum, 0, -1 / 20))
    MNfruit1max = (-1.708e-7 + 7.3125e-7 * Tcan) * 2.5
    MNbuffruit = SmoothIfElse(MCbuffruit_tot, 0.05, -58.9) * MNfruit1max
    MNbuffruit = SmoothIfElse(Tsum, 0, -1 / 20) * MNbuffruit
    logic1 = sum(h_Tcansum_Cfruit[1:-1]) == 0 or sum(Nfruit[1: -1]) == 0 or sum(GR[1: -1]) == 0
    if logic1:
        rel_factor = 0
    else:
        rel_factor = 1/(sum( GR[1:-1]*Nfruit[1:-1]))
    MCfruit = np.zeros(p['NrBox'])
    for i in np.arange(0,p['NrBox']):
        MCfruit[i] = r_dev * p['NrBox'] * Cfruit[i]
    eta_Buf_fruit = np.zeros(p['NrBox'])
    MCbuffruit = np.zeros(p['NrBox'])
    MCbuffruit[0] = MNbuffruit * Wfruit_pot[0]
    for i in np.arange(1,p['NrBox']):
        eta_Buf_fruit[i] = GR[i] * Nfruit[i] * rel_factor
        MCbuffruit[i] = eta_Buf_fruit[i] * (MCbuffruit_tot - MCbuffruit[1])
    Cstemrootdot = MCbuf_stemroot - MCstemroot_air_m
    Cleafdot = MCbufleaf - MCleafair_m - MCleafhar
    LAIdot = p['SLA'] * Cleafdot
    Cfruitdot = np.zeros(p['NrBox'])
    Nfruitdot = np.zeros(p['NrBox'])
    Cfruitdot[0] = MCbuffruit[0] - MCfruit[0] - MCfruitair_m[0]
    Nfruitdot[0] = MNbuffruit - MNfruit[0]
    for i in np.arange(1, p['NrBox']):
        Cfruitdot[i] = MCbuffruit[i] + MCfruit[i - 1] - MCfruit[i] - MCfruitair_m[i]
        Nfruitdot[i] = MNfruit[i - 1] - MNfruit[i]
    Chardot = max(0, MCfruit[-1])
    return Chardot,LAIdot,Cstemrootdot,Cleafdot,Cfruitdot,Nfruitdot