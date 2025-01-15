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
@author: Daniel Reyes Lastiri, Bo Zhou（周波）
"""
import numpy as np
from scipy.interpolate import interp1d

def fcn_euler_forward(diff,t_span,y0,h=1.0,t_eval=None,interp_kind='linear',
                      *args,**kwargs):
    """ Function for Euler Forward numerical integration.
    Based on the syntax of scipy.integrate.solve_ivp
    
    This function numerically integrates a system of ordinary differential
    equations, given an initial value::

        dy/dt = f(t, y)
        y(t0) = y0

    Here, t is a one-dimensional independent variable (time), y(t) is an
    n-dimensional vector-valued function (state), and an n-dimensional
    vector-valued function f(t, y) determines the differential equations.
    The goal is to find y(t) approximately satisfying the differential
    equations, given an initial value y(t0)=y0.
    
    Parameters
    ----------
    diff : callable
        Function to calculate the value of d_dt
        (the right-hand side of the system).
    t_span : 2-tuple of floats
        Interval of integration (t0, tf). The solver starts with t=t0 and
        integrates until it reaches t=tf.
    h : float
        Step size for the integration. It can be shorter than in
        a desired t_eval.
    t_eval : array_like or None, optional
        Times at which to store the computed solution, must be sorted and lie
        within `t_span`. If None (default), use points defined by the solver.
    interp_kind : string, optional
        Interpolation method for to computed the solution at t_eval.
        Dafault is 'linear'. Options are: 
        'linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic',
        'previous', 'next', where 'zero', 'slinear', 'quadratic' and 'cubic' 
        For details see: scipy.interpolate.interp_1d.
    
    Returns
    -------
    Dictionary with the following:\n
    t : ndarray, shape (n_points,)
        Time vector for desired evaluation (based on `t_span` and `h`,
        or equal to `t_eval`).
    y : ndarray, shape (n_states, n_points)
        Values of the solution at `t`.
    """
    nt = int((t_span[1]-t_span[0])/h) + 1
    tint = np.linspace(t_span[0],t_span[1],nt)
    yint = np.zeros((y0.size,tint.size))
    yint[:,0] = y0
    for i,ti in enumerate(tint[:-1]):
        yint[:,i+1] = y0 + diff(ti,y0)*h
        y0 = yint[:,i+1]
    if type(t_eval)=='ndarray':
        f_interp_y = interp1d(tint,yint,axis=1,
                              kind=interp_kind,fill_value='extrapolate')
        y = f_interp_y(tint)
    else:
        y = yint
    return {'t':tint, 'y':y}