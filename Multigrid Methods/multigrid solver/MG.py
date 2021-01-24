# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 13:21:37 2021

@author: mclea
"""

import numpy as np
import field as f
from scipy.interpolate import griddata


def jacobi_update(x, b, A, Rj, invD, nsteps=1, max_err=1e-3):
    """
    Solves Ax=b
    """
    # b_inner = b[1:-1, 1:-1].flatten()
    n = x.x.shape[0]
    m = x.x.shape[1]
    update=True
    step = 0
    x.set_BC()
    while update:
        step += 1
        x_old = x.x.copy()
        x_temp = Rj.dot(x.x.flatten()) + invD.dot(b.x.flatten())
        x.x = x_temp.reshape((n,m))
        x.set_BC()
        err = x.x - x_old
        if step == nsteps or np.abs(err).max()<max_err:
            update=False


def restrict(x):
    """
    Creates a new grid of points which is half the size of the original
    grid in each dimension.
    """
    n = x.x.shape[0]
    m = x.x.shape[1]
    new_n = int((n-2)/2+2)
    new_m = int((m-2)/2+2)
    new_field = f.new_field(n,m)
    new_array = np.zeros((new_n, new_m))
    for i in range(1, new_n-1):
        for j in range(1, new_m-1):
            ii = int((i-1)*2)+1
            jj = int((j-1)*2)+1
            # print(i, j, ii, jj)
            new_array[i,j] = np.average(x.x[ii:ii+2, jj:jj+2])
    new_field.x = new_array
    new_field.set_BC()
    return new_field


def interpolate_array(x):
    """
    Creates a grid of points which is double the size of the original
    grid in each dimension. Uses linear interpolation between grid points.
    """
    n = x.x.shape[0]
    m = x.x.shape[1]
    new_n = int((n-2)*2 + 2)
    new_m = int((m-2)*2 + 2)
    new_field = f.new_field(n,m)
    new_array = np.zeros((new_n, new_m))
    i = (np.indices(x.x.shape)[0]/(x.x.shape[0]-1)).flatten()
    j = (np.indices(x.x.shape)[1]/(x.x.shape[1]-1)).flatten()

    A = x.x.flatten()
    new_i = np.linspace(0, 1, new_n)
    new_j = np.linspace(0, 1, new_m)
    new_ii, new_jj = np.meshgrid(new_i, new_j)
    new_array = griddata((i, j), A, (new_jj, new_ii), method="linear")
    new_field.x = new_array
    new_field.set_BC()
    return new_field


def MGV(x, b, pm, level=0):
    """
    Solves for nabla(v) = f using a multigrid method
    """
    # print(level)
    n = x.x.shape[0]
    m = x.x.shape[1]
    # If on the largest level, find exact solution
    A, Rj, invD = pm.get_level_matrices(level)
    if level == pm.levels-2:
        jacobi_update(x, b, A, Rj, invD, nsteps=10000)
        return x
    else:
        # smoothing
        jacobi_update(x, b, A, Rj, invD, nsteps=10000, max_err=1e-1)
        
        # calculate residual
        r = f.new_field(n, m)
        r.x = (A.dot(x.x.flatten()) - b.x.flatten()).reshape(n, m)
        # print(r.x)
        zero_field = f.new_field(n,m)
        # interploate the correction computed on a corser grid
        d = interpolate_array(MGV(restrict(r), restrict(zero_field), pm, level=level+1))
        
        # Add prolongated corser grid solution onto the finer grid
        x.x = x.x - d.x
        
        jacobi_update(x, b, A, Rj, invD, nsteps=10000, max_err=1e-6)
        return x