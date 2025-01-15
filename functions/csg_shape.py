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

CSG shapes/geometry
"""

import numpy as np
import math
import scipy.optimize
from scipy import integrate
from functions import csg_fun
import time
from scipy import interpolate

class csg_shape(object):
    def __init__(self, p):
        """ Set the CSG shape parameters
        Parameters
        ----------
        p: the csg parameters
        two options to describe the CSG shape
        1. four points: A B C D
        2. diameter of each part: 'width_indoorfloor', 'height_CSG' and 'height_wall' and 'width_nroof' and 'angle_nroof'.
        At least 3 parameters are needed among ['height_CSG' , 'height_wall' , 'width_nroof' , 'angle_nroof']
        curveType - str. 'ellipse', 'parabola', 'circle', 'polyline', 'manualfunc'
                   A
                  /-）
              B /     -） g(x)
               ｜        -）
        C[0,0] ｜__________-）  D

        """
        self.p = p
        self.length_csg = 1 # 1 meter length of CSG
        self.curveType = p['curveType']
        # A B C D area_wall, are_nroof, area_floor, angle_nroof
        if 'A' and 'B' and 'C' and 'D' in p:
            self.A = p['A']
            self.B = p['B']
            self.C = p['C']
            self.D = p['D']
            self.area_wall = ((self.B[0]-self.C[0])**2+(self.B[1]-self.C[1])**2)**0.5
            self.area_nroof =((self.B[0]-self.A[0])**2+(self.B[1]-self.A[1])**2)**0.5
            self.area_floor =((self.D[0]-self.C[0])**2+(self.D[1]-self.C[1])**2)**0.5
            self.angle_nroof = round(math.atan((self.A[1]-self.B[1])/(self.A[0]-self.B[0])),3)
        elif 'height_CSG' and 'height_wall' and 'width_nroof' in p:
            self.area_wall = p['height_wall']
            self.area_nroof = p['width_nroof']
            self.area_floor = p['width_indoorfloor']
            self.A = [(p['width_nroof']**2-(p['height_CSG']-p['height_wall'])**2)**0.5,p['height_CSG']]
            self.B = [0,p['height_wall']]
            self.C = [0,0]
            self.D = [p['width_indoorfloor'],0]
            self.angle_nroof = round(math.atan((self.A[1] - self.B[1]) / (self.A[0] - self.B[0])),3)
        elif 'height_CSG' and 'height_wall' and 'angle_nroof' in p:
            self.A = [(p['height_CSG']-p['height_wall'])/math.tan(p['angle_nroof']),p['height_CSG']]
            self.B = [0,p['height_wall']]
            self.C = [0,0]
            self.D = [p['width_indoorfloor'],0]
            self.area_wall = ((self.B[0]-self.C[0])**2+(self.B[1]-self.C[1])**2)**0.5
            self.area_nroof =((self.B[0]-self.A[0])**2+(self.B[1]-self.A[1])**2)**0.5
            self.area_floor =((self.D[0]-self.C[0])**2+(self.D[1]-self.C[1])**2)**0.5
            self.angle_nroof = round(math.atan((self.A[1]-self.B[1])/(self.A[0]-self.B[0])),3)
        elif 'height_CSG' and 'angle_nroof' and 'width_nroof' in p:
            self.A = [p['width_nroof']*math.cos(p['angle_nroof']),p['height_CSG']]
            self.B = [0,p['height_CSG']-p['width_nroof']*math.sin(p['angle_nroof'])]
            self.C = [0,0]
            self.D = [p['width_indoorfloor'],0]
            self.area_wall = ((self.B[0]-self.C[0])**2+(self.B[1]-self.C[1])**2)**0.5
            self.area_nroof =((self.B[0]-self.A[0])**2+(self.B[1]-self.A[1])**2)**0.5
            self.area_floor =((self.D[0]-self.C[0])**2+(self.D[1]-self.C[1])**2)**0.5
            self.angle_nroof = round(math.atan((self.A[1]-self.B[1])/(self.A[0]-self.B[0])),3)
        else:
            self.A = [p['width_nroof']*math.cos(p['angle_nroof']),p['width_nroof']*math.sin(p['angle_nroof'])+p['height_wall']]
            self.B = [0,p['height_wall']]
            self.C = [0, 0]
            self.D = [p['width_indoorfloor'], 0]
            self.area_wall = ((self.B[0]-self.C[0])**2+(self.B[1]-self.C[1])**2)**0.5
            self.area_nroof =((self.B[0]-self.A[0])**2+(self.B[1]-self.A[1])**2)**0.5
            self.area_floor =((self.D[0]-self.C[0])**2+(self.D[1]-self.C[1])**2)**0.5
            self.angle_nroof = round(math.atan((self.A[1]-self.B[1])/(self.A[0]-self.B[0])),3)

        #  Pspg, cover_rW, cover_rH, area_cover
        if self.curveType == 'polyline':
            Pspg0 = [self.D[0]]
            Pspg0.extend(p['cover_rW'])
            Pspg0.append(self.A[0]/self.D[0])
            Pspg1 = [self.A[1]]
            Pspg1.extend(p['cover_rH'])
            Pspg1.append(self.angle_nroof)
            self.Pspg = np.array([Pspg0,Pspg1])
            self.cover_rW = p['cover_rW']
            self.cover_rH = p['cover_rH']
            self.area_cover = np.sum(((self.cover_rW*self.D[0])**2+(self.cover_rH*self.A[1])**2)**0.5)
        else:
            if 'polyline_num' in p:
                self.curveFun(p['polyline_num'])
            else:
                self.curveFun(100)  # default value is 100
        self.Vair =self.V_air()
        # view factor
        AB = self.lineLen(self.A,self.B)
        BC = self.lineLen(self.B,self.C)
        CD = self.lineLen(self.C,self.D)
        AD = self.lineLen(self.A,self.D)
        BD = self.lineLen(self.B,self.D)
        AC = self.lineLen(self.A,self.C)
        self.AVf_rofe_sky = 1
        self.AVf_rofi_cov = (AB+AD-BD)/2
        self.AVf_rofi_sky = (AB+AD-BD)/2
        self.AVf_rofi_blanki = (AB+AD-BD)/2
        self.AVf_rofi_flo = (AC+BD-BC-AD)/2
        self.AVf_rofi_wali = (AB+BC-AC)/2
        self.AVf_rofi_can = (AC+BD-BC-AD)/2
        self.AVf_wale_sky = 1
        self.AVf_wali_cov = (AC+BD-AB-CD)/2
        self.AVf_wali_sky = (AC+BD-AB-CD)/2
        self.AVf_wali_blanki = (AC+BD-AB-CD)/2
        self.AVf_wali_flo = (BC+CD-BD)/2
        self.AVf_wali_can = (BC+CD-BD)/2
        self.AVf_flo_cov = (CD+AD-AC)/2
        self.AVf_flo_sky = (CD+AD-AC)/2
        self.AVf_flo_blanki = (CD+AD-AC)/2
        self.AVf_flo_can = 1
        self.AVf_cov_can = (CD+AD-AC)/2
        self.AVf_cov_sky = 1
        self.AVf_blanki_can = (CD+AD-AC)/2
        self.AVf_blanke_sky = 1
        self.AVf_can_sky = (CD+AD-AC)/2
        # direct light tran
        self.tran_cover()





### functions -------------------------------

    def curveFun(self,polyline_num):
        self.polyline_num = polyline_num
        if self.curveType == 'ellipse':     # A and D are the two vertices of an ellipse
            self.curvefunc_str = '(x-%f)^2 / %f + y^2 / %f = 1' % (self.A[0],(self.D[0]-self.A[0])**2,self.A[1]**2)
            self.curvefunc = '(%f - %f*(x-%f)**2/%f)**0.5' % (self.A[1]**2,self.A[1]**2,self.A[0],(self.D[0]-self.A[0])**2)  # y = cervefunc
            self.gx = lambda x:(self.A[1]**2 - self.A[1]**2*(x-self.A[0])**2/(self.D[0]-self.A[0])**2)**0.5

        if self.curveType == 'circle':      # The center of the circle and point A are perpendicular to the X axis
            self.curvefunc_str = '(x-%f)^2 +(y-%f)^2 = (%f)^2' % (self.A[0],(self.A[1]**2-(self.D[0]-self.A[0])**2)/2/self.A[1],self.A[1]-(self.A[1]**2-(self.D[0]-self.A[0])**2)/2/self.A[1])
            self.curvefunc = '((%f-%f)**2-(x-%f)**2)**0.5+%f' % (self.A[1],(self.A[1]**2-(self.D[0]-self.A[0])**2)/2/self.A[1],self.A[0],(self.A[1]**2-(self.D[0]-self.A[0])**2)/2/self.A[1])
            self.gx = lambda x:((self.A[1]-(self.A[1]**2-(self.D[0]-self.A[0])**2)/2/self.A[1])**2-(x-self.A[0])**2)**0.5+(self.A[1]**2-(self.D[0]-self.A[0])**2)/2/self.A[1]

        if self.curveType == 'parabola':    # A is the vertice of parabola
            self.curvefunc_str = 'y = %f * (x-%f)^2 + %f' % (-self.A[1]/(self.D[0]-self.A[0])**2,self.A[0],self.A[1])
            self.curvefunc = '%f * (x-%f)**2 + %f' % (-self.A[1]/(self.D[0]-self.A[0])**2,self.A[0],self.A[1])
            self.gx = lambda x:-self.A[1]/(self.D[0]-self.A[0])**2* (x-self.A[0])**2 + self.A[1]

        if self.curveType == 'manualfunc':
            self.gx = self.p['curvefunc_gx']
            self.curvefunc_str = str(self.gx)

        # optimization
        rw0 = np.random.rand(polyline_num-1)*(self.D[0]-self.A[0])+self.A[0]
        cons=[]
        for consi in np.arange(0,polyline_num-1,1):
            cons.append({'type': 'ineq', 'fun': lambda x: x[consi] - self.A[0]})
            cons.append({'type':'ineq','fun':lambda x:-x[consi]+self.D[0]})

        res = scipy.optimize.minimize(self.opt_polyline, x0=rw0, method='SLSQP', constraints=cons)
        if res.success == True:
            self.opt_result=res.x
            self.opt_result.sort()
            self.cover_rW = (np.append(self.opt_result,self.D[0])-np.insert(self.opt_result,0,self.A[0]))/self.D[0]
            self.cover_rH = (np.insert(self.gx(self.opt_result),0,self.A[1])-np.append(self.gx(self.opt_result),self.D[1]))/self.A[1]
            Pspg0 = [self.D[0]]
            Pspg0.extend(self.cover_rW)
            Pspg0.append(self.A[0]/self.D[0])
            Pspg1 = [self.A[1]]
            Pspg1.extend(self.cover_rH)
            Pspg1.append(self.angle_nroof)
            self.Pspg = np.array([Pspg0,Pspg1])
            self.area_cover = np.sum(((self.cover_rW*self.D[0])**2+(self.cover_rH*self.A[1])**2)**0.5)
        else:
            print('there are bugs on optimization')
            print(res)

    def opt_polyline(self,x_w):
        """ x_w is array: X-axis coordinates of the dividing points"""
        x_w.sort()
        area_curve = integrate.quad(self.gx,self.A[0],self.D[0])
        a1_x = np.insert(x_w,0,self.A[0])
        a1_y = np.insert(self.gx(x_w),0,self.A[1])
        a2_x = np.append(x_w,self.D[0])
        a2_y = np.append(self.gx(x_w),self.D[1])
        area_line = []
        for point_i in np.arange(0, self.polyline_num, 1):
            g=self.lineFun([a1_x[point_i],a1_y[point_i]],[a2_x[point_i],a2_y[point_i]])
            area_line.append(integrate.quad(g,a1_x[point_i],a2_x[point_i])[0])
        return np.abs(area_curve[0]-np.sum(area_line))


    def lineFun(self,a1,a2):
        """give the line function between two points a1[x1,y1] and a2[x2,y2]"""
        g = lambda x:(a2[1]-a1[1])/(a2[0]-a1[0])*x+a1[1]-(a2[1]-a1[1])/(a2[0]-a1[0])*a1[0]
        return g
    def lineLen(self,a1,a2):
        """give the line length between two points a1[x1,y1] and a2[x2,y2]"""
        L = ((a1[0]-a2[0])**2+(a1[1]-a2[1])**2)**0.5
        return L
    def lineAng(self,a1,a2):
        """give the line Acute angle between line ( a1[x1,y1] and a2[x2,y2] ) and x-axis. [rad]"""
        Ang = math.atan(np.abs(a1[1]-a2[1])/np.abs(a1[0]-a2[0]))
        return Ang

    def V_air(self):
        # coordinates of points from B to D through the cover
        a_x = np.array([self.B[0],self.A[0]])
        self.a_x = np.append(a_x,np.cumsum(self.cover_rW*self.D[0])+self.A[0])
        a_y = np.array([self.B[1],self.A[1]])
        self.a_y = np.append(a_y,self.A[1]-np.cumsum(self.cover_rH*self.A[1]))
        area_line = []
        for point_i in np.arange(0,len(self.cover_rW)+1,1):
            g = self.lineFun([self.a_x[point_i], self.a_y[point_i]], [self.a_x[point_i+1], self.a_y[point_i+1]])
            area_line.append(integrate.quad(g, self.a_x[point_i], self.a_x[point_i+1])[0])
        return np.sum(area_line)


    def tran_cover(self):
        if 'sim_t_tran' in self.p:
            dt = self.p['sim_t_tran']
        else:
            dt = 600  # default value(600s) for light transmission time difference
        t1 = time.strptime(self.p['StartTime'],'%Y-%m-%dT%H:%M')
        t2 = time.strptime(self.p['EndTime'],'%Y-%m-%dT%H:%M')
        end_t = ((t2.tm_year-t1.tm_year)*365+t2.tm_yday-t1.tm_yday)*86400+(t2.tm_hour+t2.tm_min/60-t1.tm_hour-t1.tm_min/60)*3600
        self.sim_t = np.arange(0,end_t,dt)
        cover = self.cover()# np.array([[slop(rad)], [length(m)]]) ??
        angle_alt=np.array([])
        tran_cover = np.array([])
        for ti in self.sim_t:
            angle_solar = csg_fun.solarAlt(ti,self.p['StartTime'],[self.p['latitude'],self.p['longitude'],self.p['Hourcircle']])
            angle_alt=np.append(angle_alt,angle_solar[0])
            tran_1 = csg_fun.directtran(angle_solar,self.p['Angle_CSG'],cover,[self.p['n'],self.p['thickness_cover'],self.p['C_abs']])
            tran_cover=np.append(tran_cover,tran_1)
        self.angle_alt = abs(angle_alt.round(3))
        self.tran_cov = tran_cover.round(3)
        self.f_Tran = interpolate.interp1d(self.sim_t, self.tran_cov, kind='slinear',
                                               fill_value='extrapolate')
        self.f_alt  = interpolate.interp1d(self.sim_t, self.angle_alt, kind='slinear',
                                               fill_value='extrapolate')


    def cover(self):
        l1=[]
        l2=[]
        for i in np.arange(1,len(self.a_x)-1,1):
            l1.append(self.lineAng([self.a_x[i],self.a_y[i]],[self.a_x[i+1],self.a_y[i+1]]))
            l2.append(self.lineLen([self.a_x[i],self.a_y[i]],[self.a_x[i+1],self.a_y[i+1]]))

        cover = np.array([l1,l2])
        return cover


