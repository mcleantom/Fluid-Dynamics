# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 16:24:16 2020

@author: mclea
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from matplotlib import cm
import kernprof
from line_profiler import LineProfiler
from scipy.sparse import csr_matrix
import time

def restrict(A):
    """
    Creates a new grid of points which is half the size of the original
    grid in each dimension.
    """
    n = A.shape[0]
    m = A.shape[1]
    new_n = int((n-2)/2+2)
    new_m = int((m-2)/2+2)
    new_array = np.zeros((new_n, new_m))
    for i in range(1, new_n-1):
        for j in range(1, new_m-1):
            ii = int((i-1)*2)+1
            jj = int((j-1)*2)+1
            # print(i, j, ii, jj)
            new_array[i,j] = np.average(A[ii:ii+2, jj:jj+2])
    new_array = set_BC(new_array)
    return new_array


def interpolate_array(A):
    """
    Creates a grid of points which is double the size of the original
    grid in each dimension. Uses linear interpolation between grid points.
    """
    n = A.shape[0]
    m = A.shape[1]
    new_n = int((n-2)*2 + 2)
    new_m = int((m-2)*2 + 2)
    new_array = np.zeros((new_n, new_m))
    i = (np.indices(A.shape)[0]/(A.shape[0]-1)).flatten()
    j = (np.indices(A.shape)[1]/(A.shape[1]-1)).flatten()

    A = A.flatten()
    new_i = np.linspace(0, 1, new_n)
    new_j = np.linspace(0, 1, new_m)
    new_ii, new_jj = np.meshgrid(new_i, new_j)
    new_array = griddata((i, j), A, (new_jj, new_ii), method="linear")
    return new_array


def adjacency_matrix(rows, cols):
    """
    Creates the adjacency matrix for an n by m shaped grid
    """
    n = rows*cols
    M = np.zeros((n,n))
    for r in range(rows):
        for c in range(cols):
            i = r*cols + c
            # Two inner diagonals
            if c > 0: M[i-1,i] = M[i,i-1] = 1
            # Two outer diagonals
            if r > 0: M[i-cols,i] = M[i,i-cols] = 1
    return M


def create_differences_matrix(rows, cols):
    """
    Creates the central differences matrix A for an n by m shaped grid
    """
    n = rows*cols
    M = np.zeros((n,n))
    for r in range(rows):
        for c in range(cols):
            i = r*cols + c
            # Two inner diagonals
            if c > 0: M[i-1,i] = M[i,i-1] = -1
            # Two outer diagonals
            if r > 0: M[i-cols,i] = M[i,i-cols] = -1
    np.fill_diagonal(M, 4)
    return M


def set_BC(A):
    """
    Sets the boundary conditions of the field
    """
    A[:, 0] = A[:, 1]
    A[:, -1] = A[:, -2]
    A[0, :] = A[1, :]
    A[-1, :] = A[-2, :]
    return A


def create_A(n,m):
    """
    Creates all the components required for the jacobian update function
    for an n by m shaped grid
    """
    LaddU = adjacency_matrix(n,m)
    A = create_differences_matrix(n,m)
    invD = np.zeros((n*m, n*m))
    np.fill_diagonal(invD, 1/4)
    return A, LaddU, invD


def calc_RJ(rows, cols):
    """
    Calculates the jacobian update matrix Rj for an n by m shaped grid
    """
    n = int(rows*cols)
    M = np.zeros((n,n))
    for r in range(rows):
        for c in range(cols):
            i = r*cols + c
            # Two inner diagonals
            if c > 0: M[i-1,i] = M[i,i-1] = 0.25
            # Two outer diagonals
            if r > 0: M[i-cols,i] = M[i,i-cols] = 0.25

    return M


def create_jacobi_update_arrays(n,m):
    A, LaddU, invD = create_A(n,m)
    Rj = calc_RJ(n,m)
    A = csr_matrix(A)
    LaddU = csr_matrix(LaddU)
    invD = csr_matrix(invD)
    Rj = csr_matrix(Rj)
    return A, Rj, invD


def jacobi_update2(v, f, A, Rj, invD, nsteps=1, max_err=1e-3):
    """
    """
    f_inner = f[1:-1, 1:-1].flatten()
    n = v.shape[0]
    m = v.shape[1]
    update=True
    step = 0
    v.set_BC()
    while update:
        v_old = v.copy()
        step += 1
        vt = v_old[1:-1, 1:-1].flatten()
        vt = Rj.dot(vt) + invD.dot(f_inner)
        v[1:-1, 1:-1] = vt.reshape((n-2),(m-2))
        err = v - v_old
        if step == nsteps or np.abs(err).max()<max_err:
            update=False
    
    return v, (step, np.abs(err).max())


def jacobi_update(v, f, nsteps=1, max_err=1e-3):
    """
    Uses a jacobian update matrix to solve nabla(v) = f
    """
    
    f_inner = f[1:-1, 1:-1].flatten()
    n = v.shape[0]
    m = v.shape[1]
    A, LaddU, invD = create_A(n-2, m-2)
    Rj = calc_RJ(n-2,m-2)
    A = csr_matrix(A)
    LaddU = csr_matrix(LaddU)
    invD = csr_matrix(invD)
    Rj = csr_matrix(Rj)
    
    update=True
    step = 0
    while update:
        v_old = v.copy()
        step += 1
        vt = v_old[1:-1, 1:-1].flatten()
        vt = Rj.dot(vt) + invD.dot(f_inner)
        v[1:-1, 1:-1] = vt.reshape((n-2),(m-2))
        err = v - v_old
        if step == nsteps or np.abs(err).max()<max_err:
            update=False
    
    return v, (step, np.abs(err).max())


def MGV(f, v):
    """
    Solves for nabla(v) = f using a multigrid method
    """
    # global  A, r
    n = v.shape[0]
    m = v.shape[1] 
    
    # If on the smallest grid size, compute the exact solution
    if n <= 6 or m <=6:
        v, info = jacobi_update(v, f, nsteps=1000)
        return v
    else:
        # smoothing
        v, info = jacobi_update(v, f, nsteps=10, max_err=1e-1)
        A = create_A(n, m)[0]
        
        # calculate residual
        r = np.dot(A, v.flatten()) - f.flatten()
        r = r.reshape(n,m)
        
        # downsample resitdual error
        r = restrict(r)
        zero_array = np.zeros(r.shape)
        
        # interploate the correction computed on a corser grid
        d = interpolate_array(MGV(r, zero_array))
        
        # Add prolongated corser grid solution onto the finer grid
        v = v - d
        
        v, info = jacobi_update(v, f, nsteps=10, max_err=1e-6)
        return v


sigma = 0

# Setting up the grid
k = 7
n = 2**k+2
m = 2**(k)+2

hx = 1/n
hy = 1/m

L = 1
H = 1

x = np.linspace(0, L, n)
y = np.linspace(0, H, m)
XX, YY = np.meshgrid(x, y)

# Setting up the initial conditions
f = np.ones((n,m))
f[1:int(m/2), 1:int(n/2)] = -1
v = np.zeros((n,m))

# How many V cyles to perform
err = 1
n_cycles = 10
loop = True
cycle = 0

start = time.time()
A, Rj, invD = create_jacobi_update_arrays(n-2, m-2)
end = time.time()
print(end - start)

# lp = LineProfiler()
# lp_wrapper = lp(jacobi_update2)
# lp_wrapper(v, f, A, Rj, invD, nsteps=100)
# lp.print_stats()

start = time.time()
u = jacobi_update2(v, f, A, Rj, invD, nsteps=10000)
end = time.time()
print(end - start)
print(str(u[1][0]) + " loops")
print(str(u[1][1]) + " Error")

# u = MGV(f, v)

# Perform V cycles until converged or reached the maximum
# number of cycles
# while loop:
#     cycle += 1
#     v_new = MGV(f, v)
    
#     if np.abs(v - v_new).max() < err:
#         loop = False
#     if cycle == n_cycles:
#         loop = False
    
#     v = v_new

# print("Number of cycles " + str(cycle))
# plt.contourf(v)