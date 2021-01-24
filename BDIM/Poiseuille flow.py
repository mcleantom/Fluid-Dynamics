# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 14:44:55 2020

@author: mclea
"""

import numpy as np
import matplotlib.pyplot as plt

f = 1
v = 1
L = 1
N = int(1000)
dy = L / N
Y = np.linspace(0, L, N)
epsilon = 0.1
c = dy**2
U = 0

def calc_distance(y, L):
    """
    Calculates the signed distance of a given array of points from the nearest
    point inbetween two flat plates at y=0 and y=L.

    Parameters
    ----------
    y : numpy array
        An array of points along a line.
    L : float
        The distance between two plates in a Poiseuille flow. One plate is at
        y=0 and the other is at y=L.

    Returns
    -------
    distance: numpy array
        Returns the signed distance to the nearest point between two flat
        plates, chosen negative beyond the flat plates.

    """
    distance = L/2 - np.abs(L/2 - y)
    return distance

def calc_delta_solid_body(d, epsilon):
    """
    Calculates the interpolation function delta for a immersed solid body

    Parameters
    ----------
    d : numpy array
        The signed distance to the nearest point on a fluid/solid interface,
        chosen negative within the solid.
    epsilon : float
        The kernal diameter.

    Returns
    -------
    delta : numpy array
        Returns the kernal zeroth moment over a body, delta epsilon B. The
        integrated value is 1 within the body, 0 on the exterior and a smooth
        transition over 2*epsilon.

    """
    in_kernal = abs(d)/epsilon < 1
    outside_kernal = d/epsilon < -1
    delta = np.zeros(d.shape)
    delta[in_kernal] = 0.5*(1-np.sin((np.pi/2)*(d[in_kernal]/epsilon)))
    delta[outside_kernal] = 1
    return delta

def calc_delta_surface(d, epsilon):
    """
    Calculates the interpolation function delta for an immersed solid surface

    Parameters
    ----------
    d : numpy array
        The signed distance to the nearest point on a fluid/solid interface,
        chosen negative within the solid.
    epsilon : TYPE
        The kernal diameter.

    Returns
    -------
    delta: numpy array
        Returns the kernal zeroth moment over a immersed surface,
        delta epsilon S. The integrated value is smooth within a kernal distance
        2*epsilon and 0 elsewhere.
    """
    in_kernal = abs(d)/epsilon < 1
    delta = np.zeros(d.shape)
    delta[in_kernal] = 0.5*(1+np.cos(np.pi*(d[in_kernal]/epsilon)))
    
    return np.vstack(delta)

def plot_delta(y, epsilon, L):
    Y = np.linspace(-0.5, 1.5, 1000)
    distance = calc_distance(Y, L)
    delta1 = calc_delta_solid_body(distance, epsilon)
    delta2 = calc_delta_surface(distance, epsilon)
    plt.figure()
    plt.plot(Y, distance)
    plt.plot(Y, delta1)
    plt.plot(Y, delta2)

# plot_delta(Y, epsilon, L)

Dij = np.zeros((N,N))
np.fill_diagonal(Dij, -2/dy**2)
np.fill_diagonal(Dij[1:], 1/dy**2)
np.fill_diagonal(Dij[:,1:], 1/dy**2)
N = Dij*dy**2

distance = calc_distance(Y, L)
delta = calc_delta_surface(distance, epsilon)

d = np.zeros((len(delta), len(delta)))
np.fill_diagonal(d, delta)

x = -1*(1-delta)*(Dij*c)

A = (1/dy**2)*(d + x)
B = (f/v)*(1-delta)

U = np.linalg.solve(A, B)
Umax = (f*L**2)/(8*v)
UoUmax = U/Umax

plt.plot(Y, UoUmax)
plt.xlim(0, 1)
plt.ylim(0, 1)

E = (max(U)-Umax)
print(abs(E)/Umax)