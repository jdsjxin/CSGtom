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
FTE34806 - Modelling of Biobased Production Systems
Agricultural Biosystems Engineering Group, WUR
@authors: Daniel Reyes Lastiri, Rachel van Ooteghem, Tim Hoogstad, Bo Zhou（周波）

Basic class for models
"""

import copy
import numpy as np
import pandas as pd

class Module():
    def __init__(self,tsim,dt,x0,p):
        self.tsim = tsim
        dtsim = (self.tsim[1]-self.tsim[0])
        self.dt = dt
        len_t = int(dtsim/dt*(len(tsim)-1) + 1)
        self.t = np.linspace(self.tsim[0],self.tsim[-1],len_t)
        if x0:
            self.x0 = copy.deepcopy(x0)
        else: self.x0 = None
        if p:
            self.p = copy.deepcopy(p)
        else: self.p = None
        self.y = {}
    
    def run(self,tspan,d=None,u=None):
        """ Run the attribute 'model' of Module instance
        
        Parameters
        ----------
        tspan : 2-element array-like
            initial and final time for the model run
        d : dictionary
            disturbances
        u : dictionary
            controlled inputs
        """
        # Call model
        y = self.model(tspan,self.dt,self.x0,self.p,d,u)
        # Update model output logs
        # (if first simulation, initialize logs)
        if len(self.y) == 0:
            self.y_keys = [yk for yk in y.keys() if yk!='t']
            for k in self.y_keys:
                self.y[k] = np.full((len(self.t),),np.nan)
        # (else, update)
        else:
            for k in self.y_keys:
                idxs = np.isin(self.t,y['t'])
                self.y[k][idxs] = y[k]
        # Update initial conditions
        for k in self.x0.keys():
            self.x0[k] = y[k][-1]
        return y

    def ns(self,x0,p_ref,d=None,u=None,y_keys=None):
        # Reset intial conditions for reference module
        self.x0 = copy.deepcopy(x0)
        # Initialize instances for -/+
        instance_mns = copy.deepcopy(self)
        instance_pls = copy.deepcopy(self)
        # Run module for reference parameters
        tspan = (self.tsim[0],self.tsim[-1])
        y_ref = self.run(tspan,d,u)
        # Initialize pandas dataframe for normalized sensitivities (with nan)
        # - create MultiIndex
        if not y_keys:
            y_keys = [yk for yk in self.y_keys]
        p_keys = [pk for pk in p_ref.keys()]
        pm = ['-','+']
        iterables = [y_keys,p_keys,pm]
        midx = pd.MultiIndex.from_product(iterables, names=['y', 'p', '-/+'])
        # - initialize nan DataFrame
        nan_arr = np.full((self.t.size, 2*len(y_keys)*len(p_keys)),0.)
        ns_df = pd.DataFrame(nan_arr, index=self.t, columns=midx)
        # Iterate over model parameters
        for kp in p_ref.keys():
            # Reset initial conditions and parameters to for mns and pls modules
            instance_mns.x0 = copy.deepcopy(x0)
            instance_pls.x0 = copy.deepcopy(x0)
            p_mns = copy.deepcopy(self.p)
            p_pls = copy.deepcopy(self.p)
            instance_mns.p = p_mns
            instance_pls.p = p_pls
            # Modify parameter kp
            instance_mns.p[kp] = 0.95*p_ref[kp]
            instance_pls.p[kp] = 1.05*p_ref[kp]
            # Model reset
            instance_mns.resetModel()
            instance_pls.resetModel()

            if kp[0] == 'T':
                instance_mns.p[kp] = p_ref[kp]-1.0
                instance_pls.p[kp] = p_ref[kp]+1.0
            # Run model
            y_mns = instance_mns.run(tspan,d,u)
            y_pls = instance_pls.run(tspan,d,u)
            # Compute normalized sensitivities per model output
            for ky in y_keys:
                # Compute sensitivity
                s_mns = (y_mns[ky] - y_ref[ky]) / (-0.05)
                s_pls = (y_pls[ky] - y_ref[ky]) / (0.05)
                # Compute normalized sensitivity
                ns_mns = s_mns / np.average(y_ref[ky])
                ns_pls = s_pls / np.average(y_ref[ky])
                # Store in corresponding column in DataFrame
                # (tuple to access MultiIndex column name)
                ns_df.loc[:,(ky,kp,'-')] = ns_mns
                ns_df.loc[:,(ky,kp,'+')] = ns_pls
        # Add columns for reference model outputs
        for yk in y_keys:
            ns_df[yk,'ref','ref'] = y_ref[yk]
        # Results
        return ns_df
        
            

