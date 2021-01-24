# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 15:49:34 2020

@author: mclea
"""
import numpy as np
from scipy import interpolate
from skimage.measure import block_reduce
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

from scipy import ndimage

def grad(A, dx, dy):
    return np.gradient(A, dx, dy)

def div(A, dx, dy):
    return np.sum(grad(A, dx, dy), axis=0)

def block_mean(ar, fact):
    assert isinstance(fact, int), type(fact)
    sx, sy = ar.shape
    X, Y = np.ogrid[0:sx, 0:sy]
    regions = sy/fact * (X/fact) + Y/fact
    res = ndimage.mean(ar, labels=regions, index=np.arange(regions.max() + 1))
    res.shape = (sx/fact, sy/fact)
    return res


def field(n):
    return 


def jacobi_step(v, f):
    
    # How many rows and columns
    m, n = v.shape
    
    v_update = v.copy()
    
    # for every point that isnt on the boundary
    for i in range(1, m-1):
        for j in range(1, n-1):
            v_update[i,j] = (v[i+1,j] + v[i-1,j] + v[i,j-1] + v[i, j+1] + f[i,j])/4
    
    return v_update


def restrict(A):
    n = A.shape[0]
    m = A.shape[1]
    new_n = int((n-2)/2+2)
    new_m = int((m-2)/2+2)
    new_array = np.zeros((new_n, new_m))
    for i in range(1, new_n-1):
        for j in range(1, new_m-1):
            ii = int((i-1)*2)+1
            jj = int((j-1)*2)+1
            print(i, j, ii, jj)
            new_array[i,j] = np.average(A[ii:ii+2, jj:jj+2])
    new_array = set_BC(new_array)
    return new_array


def interpolate_array(A):
    n = A.shape[0]
    m = A.shape[1]
    new_n = int((n-2)*2 + 2)
    new_m = int((m-2)*2 + 2)
    new_array = np.zeros((new_n, new_m))
    i = (np.indices(A.shape)[0]/(A.shape[0]-1)).flatten()
    j = (np.indices(A.shape)[1]/(A.shape[1]-1)).flatten()
    # points = np.vstack((i.flatten(), j.flatten())).T
    A = A.flatten()
    new_i = np.linspace(0, 1, new_n)
    new_j = np.linspace(0, 1, new_m)
    new_ii, new_jj = np.meshgrid(new_i, new_j)
    new_array = griddata((i, j), A, (new_jj, new_ii), method="linear")
    return new_array
    

def set_BC(A):
    A[:, 0] = 0
    A[:, -1] = 0
    A[0, :] = 0
    A[-1, :] = 0
    return A
    

# def MGV(f, v):
#     n = f.shape[0]
#     m = f.shape[1]
#     if m<=4 or n<=4:
#         loop = True
#         while True:
#             v = jacobi_step(v, f)
#             err = 
#         return 


k = 5
n = 2**k+2
m = 2**(k)+2

hx = 1/n
hy = 1/m

L = 1
H = 1

x = np.linspace(0, L, n)
y = np.linspace(0, H, m)
XX, YY = np.meshgrid(x, y)

p = np.ones((n, m))
# p = np.sin(XX)
# p = set_BC(p)
v = np.zeros((n, m))

nsteps = 1000

plt.figure()
plt.contourf(v)
for i in range(nsteps):
    v = jacobi_step(v, p)

plt.contourf(v)