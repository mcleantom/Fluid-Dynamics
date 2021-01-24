# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 14:37:56 2020

@author: mclea
"""

import numpy as np
import matplotlib.pyplot as plt
import math

def grid_with_boundary(k):
    """
    

    Parameters
    ----------
    k : int
        Creates a grid of size 2^k.

    Returns
    -------
    f : numpy array
        The value of the grid with the boundary conditions
        applied.

    """

    n = 2**k

    # grid
    f = np.zeros((n+1, n+1))

    # boundary conditions

    # bottom
    f[0, 0:(int(n/2))] = np.linspace(13, 5, int((n/2)))
    f[0, int(n/2):int(3*n/4)] = 5
    f[0, int(3*n/4):-1] = np.linspace(5, 13, int(n/4))

    # top
    f[-1, :] = 21

    # left
    f[0:int(3*n/8), 0] = np.linspace(13, 40, int(3*n/8))
    f[int(n/2):-1, 0] = np.linspace(40, 21, int(n/2))

    # right
    f[0:int(n/2), -1] = np.linspace(13, 40, int(n/2))
    f[int(5*n/8):-1, -1] = np.linspace(40, 21, int(3*n/8))

    # heaters
    f[int(3*n/8):int(n/2), 0:int(n/8+1)] = 40
    f[int(n/2):int(5*n/8), int(7*n/8):(n+1)] = 40
    return f

def add_heaters(T):
    m, n = T.shape
    T_update = T.copy()
    T_update[int(3*n/8):int(n/2), 0:int(n/8+1)] = 40
    T_update[int(n/2):int(5*n/8), int(7*n/8):(n+1)] = 40
    return T_update

def simulation(T, e=1E-3, num_steps=200):
    
    results = [T]
    m, n = T.shape
    
    for i in range(num_steps):
        
        T_update = jacobi_step(T)
        
        T_update = add_heaters(T_update)
        
        results.append(T_update.copy())
        
        if np.max(np.abs(T_update - T)) < e:
            return results
        
        T = T_update.copy()

    return results

def jacobi_step(T):
    
    # How many rows and columns
    m, n = T.shape
    
    T_update = T.copy()
    
    # for every point that isnt on the boundary
    for i in range(1, m-1):
        for j in range(1, n-1):
            T_update[i,j] = (T[i+1,j] + T[i-1,j] + T[i,j-1] + T[i, j+1])/4
    
    return T_update

def initial_condition(T):
    
    m, n = T.shape
    
    T_update = T.copy()
    
    for i in range(1, m-1):
        for j in range(1, n-1):
            T_update[i,j] = (j*T[i, -1] + (n+1-j)*T[i, 0] +
                             (m+1-i)*T[0, j] + m*T[-1, j])/(m+n+2)
    
    return T_update

def uplevel(T):
    
    _, n = T.shape
    
    # Workout what k was used
    k = int(np.log2(n))
    
    T_update = grid_with_boundary(k+1)
    
    T_update = add_heaters(T_update)
    _, new_n = T_update.shape
    
    for i in range(1, new_n-1):
        for j in range(1, new_n-1):
            T_update[i,j] = T[int(i/2), int(j/2)]
    
    return T_update

def multilevel(T, k_final, e =1e-2, num_steps=5000):
    
    _, n = T.shape
    k_start = int(np.log2(n))
    
    ks = []
    iterations = []
    
    for k in range(k_start, k_final+1):
        ks.append(k)
        
        results = simulation(T, e=e, num_steps=num_steps)
        iterations.append(len(results))
        
        T = results[-1]
        
        if k < k_final:
            T = uplevel(T)
            T = add_heaters(T)
    
    return T, ks, iterations

starting_point = grid_with_boundary(5)

# T_end, ks_end, its = multilevel(starting_point, 10)
# plt.imshow(T_end)