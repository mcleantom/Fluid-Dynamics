# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 14:14:34 2020

@author: mclea
"""
import numpy as np
import matplotlib.pyplot as plt

dx = 0.01
X = np.arange(-4, 4+dx, dx)
Y = np.arange(-4, 4+dx, dx)
XY = np.meshgrid(X, Y)
a = 1
epsilon = 0.02
N =  np.array([[1,0],
               [0,1]])

def cartesian_to_polar(x, y):
    """
    

    Parameters
    ----------
    X : Numpy array
        A numpy array of X coordinates
    Y : Numpy array
        A numpy array of Y coordinats

    Returns
    -------
    r : The radial distance
    theta : The angle, in radians

    """
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan(y/x)
    return (r, theta)

def calc_nearest_point(r, theta, a):
    """
    

    Parameters
    ----------
    r : Numpy array
        The radial distance
    theta : Numpy array
        The angle, in radians
    a : float
        The diameter of the circle

    Returns
    -------
    None.

    """
    new_r = np.ones(r.shape)*a
    return (new_r, theta)

def calc_distance(original, closest):
    """
    

    Parameters
    ----------
    original : Numpy array
        The coordinates where you want to know the distance to the closest surface
    closest : TYPE
        The closest point to the points of interest

    Returns
    -------
    distance : Numpy array
        The distance to the nearest surface, negative within the body.

    """
    distance = original[0]-closest[0]
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

def divergence(f):
    """
    Calculates the divergence of a field f(x, y, z,...)

    Parameters
    ----------
    f : Numpy array
        A numpy array of a vector field [U, V, W]

    Returns
    -------
    divergence:
       The divergence of the vector field [U, V, W]

    """
    num_dims = len(f)
    return np.ufunc.reduce(np.add, [np.gradient(f[i], axis=i) for i in range(num_dims)])

U = -1*np.ones(XY[0].shape) # Uniform flow of magnitude 1
V = 0*np.ones(XY[0].shape) # Zero flow up
velocity = np.array([U, V])

polar_coords = cartesian_to_polar(XY[0], XY[1])
nearest_coords = calc_nearest_point(polar_coords[0], polar_coords[1], a)
distance = calc_distance(polar_coords, nearest_coords)
delta = calc_delta_solid_body(distance, epsilon)

velocity = delta*velocity
g = divergence(velocity)

# uu = delta * U
# B = np.diff(delta*U, axis=1)
# plt.figure()
plt.contourf(XY[0], XY[1], g, cmap="RdGy")
plt.colorbar();