# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 19:43:09 2020

@author: mclea
"""

import numpy as np
import matplotlib.pyplot as plt

L = 2
epsilon = 0.2
nx = 100
ny = 100
nt = 10
nit = 100
Uwall = 0
F = 1
dx = 2.0/(nx-1)
dy = 2.0/(ny-1)
x = np.linspace(0,L,nx)
y = np.linspace(-L/2, 1.5*L,ny)
X,Y = np.meshgrid(x,y)

rho = 1
nu = 0.1
dt = .002

u = np.ones((ny, nx))
v = np.zeros((ny, nx))
p = np.zeros((ny, nx)) 
b = np.zeros((ny, nx))

def presPoisson(p, dx, dy,rho,nu,u,v):
    pn = np.empty_like(p)
    pn = p.copy()
     
    #Term in square brackets
    b[1:-1,1:-1]=rho*(1/dt*((u[1:-1,2:]-u[1:-1,0:-2])/(2*dx)+(v[2:,1:-1]-v[0:-2,1:-1])/(2*dy))-\
                 ((u[1:-1,2:]-u[1:-1,0:-2])/(2*dx))**2-\
                 2*((u[2:,1:-1]-u[0:-2,1:-1])/(2*dy)*(v[1:-1,2:]-v[1:-1,0:-2])/(2*dx))-\
                 ((v[2:,1:-1]-v[0:-2,1:-1])/(2*dy))**2)

    for q in range(nit):
        pn = p.copy()
        p[1:-1,1:-1] = ((pn[1:-1,2:]+pn[1:-1,0:-2])*dy**2+(pn[2:,1:-1]+pn[0:-2,1:-1])*dx**2)/\
                        (2*(dx**2+dy**2)) -\
                        dx**2*dy**2/(2*(dx**2+dy**2))*b[1:-1,1:-1]
        
        p[-1,:] = p[-2,:] #dp/dy = 0 at y = 2
        p[0,:] = p[1,:]  #dp/dy = 0 at y = 0
        p[:,0] = p[:,1]  #dp/dx = 0 at x = 0
        p[:,-1] = p[:,-2] #dp/dx=0 at x=2
        p[0,0] = 0   #initalize the pressure     
        
    return p


def calc_distance(X, Y, L):
    """
    Calculates the distance from the nearest wall from y=0 to y=L

    Parameters
    ----------
    X : TYPE
        DESCRIPTION.
    Y : TYPE
        DESCRIPTION.
    L : TYPE
        DESCRIPTION.

    Returns
    -------
    Distance: numpy array

    """
    distance = L/2 - np.abs(L/2-Y)
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


def cavityFlow(nt, u, v, dt, dx, dy, p, rho, nu, delta, Uwall, F):
    un = np.empty_like(u)
    vn = np.empty_like(v)
    
    
    for n in range(nt):
        un = u.copy()
        vn = v.copy()
        
        p = presPoisson(p, dx, dy, rho, nu, u, v)

        u[1:-1,1:-1] = un[1:-1,1:-1]-\
                        un[1:-1,1:-1]*dt/dx*(un[1:-1,1:-1]-un[1:-1,0:-2])-\
                        vn[1:-1,1:-1]*dt/dy*(un[1:-1,1:-1]-un[0:-2,1:-1])-\
                        dt/(2*rho*dx)*(p[1:-1,2:]-p[1:-1,0:-2])+\
                        nu*(dt/dx**2*(un[1:-1,2:]-2*un[1:-1,1:-1]+un[1:-1,0:-2])+\
                        dt/dy**2*(un[2:,1:-1]-2*un[1:-1,1:-1]+un[0:-2,1:-1]))

        v[1:-1,1:-1] = vn[1:-1,1:-1]-\
                        un[1:-1,1:-1]*dt/dx*(vn[1:-1,1:-1]-vn[1:-1,0:-2])-\
                        vn[1:-1,1:-1]*dt/dy*(vn[1:-1,1:-1]-vn[0:-2,1:-1])-\
                        dt/(2*rho*dy)*(p[2:,1:-1]-p[0:-2,1:-1])+\
                        nu*(dt/dx**2*(vn[1:-1,2:]-2*vn[1:-1,1:-1]+vn[1:-1,0:-2])+\
                        (dt/dy**2*(vn[2:,1:-1]-2*vn[1:-1,1:-1]+vn[0:-2,1:-1])))
        
        # u -= delta*(-1*Uwall)
        
        u[0,:] = u[1,:]
        u[:,0] = u[:,1]
        u[:,-1] = u[:,-2]
        u[-1,:] = u[-2,:]
        v[0,:] = v[1,:]
        v[-1,:]=  v[-2,:]
        v[:,0] = v[:,1]
        v[:,-1] = v[:,-2]
        # u = u - delta*Uwall
        
        F = u
        B = (Uwall - u)
        
        u = (1-delta)*F + delta*B

    return u, v, p

def plot_flow(X, Y, u, v):
    """
    """
    fig = plt.figure()
    # plt.contourf(X, Y, u)
    plt.quiver(X, Y, u, v)


distance = calc_distance(X, Y, L)
delta = calc_delta_solid_body(distance, epsilon)
# u = u - delta*Uwall

u, v, p = cavityFlow(nt, u, v, dt, dx, dy, p, rho, nu, delta, Uwall, F)

# p = presPoisson(p, dx, dy, rho, nu, u, v)

plot_flow(X, Y, u, v)

fig, ax = plt.subplots()
cs = ax.contourf(X, Y, p)
fig.colorbar(cs, ax=ax, shrink=0.9)
plt.show()
