# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 17:19:17 2020

@author: mclea
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Function to solve Possion equation
def presPoisson(p, dx, dy,rho):
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

def cavityFlow(nt, u, v, v0, dt, dx, dy, p, rho, nu, D, T, T0, T_high, g, beta):
    un = np.empty_like(u)
    vn = np.empty_like(v)
    b = np.zeros((ny, nx))
    
    for n in range(nt):
        un = u.copy()
        vn = v.copy()
        Tn = T.copy()
        
        #compute p
        p = presPoisson(p, dx, dy, rho)
        
        #compute u
        u[1:-1,1:-1] = un[1:-1,1:-1]-\
                        un[1:-1,1:-1]*dt/dx*(un[1:-1,1:-1]-un[1:-1,0:-2])-\
                        vn[1:-1,1:-1]*dt/dy*(un[1:-1,1:-1]-un[0:-2,1:-1])-\
                        dt/(2*rho*dx)*(p[1:-1,2:]-p[1:-1,0:-2])+\
                        nu*(dt/dx**2*(un[1:-1,2:]-2*un[1:-1,1:-1]+un[1:-1,0:-2])+\
                        (dt/dy**2*(un[2:,1:-1]-2*un[1:-1,1:-1]+un[0:-2,1:-1])))

        #compute v
        v[1:-1,1:-1] = vn[1:-1,1:-1]-\
                        un[1:-1,1:-1]*dt/dx*(vn[1:-1,1:-1]-vn[1:-1,0:-2])-\
                        vn[1:-1,1:-1]*dt/dy*(vn[1:-1,1:-1]-vn[0:-2,1:-1])-\
                        dt/(2*rho*dy)*(p[2:,1:-1]-p[0:-2,1:-1])+\
                        nu*(dt/dx**2*(vn[1:-1,2:]-2*vn[1:-1,1:-1]+vn[1:-1,0:-2])+\
                        (dt/dy**2*(vn[2:,1:-1]-2*vn[1:-1,1:-1]+vn[0:-2,1:-1])))-(1-beta*(Tn[1:-1,1:-1]-T0))*g*dt
                        
                        
        #compute T
        T[1:-1,1:-1] = Tn[1:-1,1:-1]+\
                       D*dt*((Tn[2:,1:-1]-2*Tn[1:-1,1:-1]+Tn[:-2,1:-1])/(dx**2)+\
                             (Tn[1:-1,2:]-2*Tn[1:-1,1:-1]+Tn[1:-1,:-2])/(dy**2))-\
                       (un[1:-1,1:-1]*(Tn[2:,1:-1]-Tn[:-2,1:-1])/(2*dx)+vn[1:-1,1:-1]*(Tn[1:-1,2:]-Tn[1:-1,:-2])/(2*dy))*dt-\
                       Tn[1:-1,1:-1]*((un[2:,1:-1]-un[:-2,1:-1])/(2*dx)+Tn[1:-1,1:-1]*(vn[1:-1,2:]-vn[1:-1,:-2]))/(2*dy)*dt
        
        
        #Temperature is pinned at top and bottom surfaces. Open boundary condition is applied at the side.
        T[0,:] = T_high
        T[:,0] = 2*T[:,1]-T[:,2]
        T[:,-1] = 2*T[:,-2]-T[:,-3]
        T[-1,:] = T0
        
        #No-slip boudary condition is applied at all boundary
        u[0,:] = 0
        u[:,0] = 0
        u[:,-1] = 0
        u[-1,:] = 0
        v[0,:] = 0.01*n*v0[:]*np.exp(-0.01*n)
        v[-1,:] = 0
        v[:,0] = 0
        v[:,-1] = 0
        
        
    return u, v, p, T

# Function to plot graph
def makegraph():
    plt.contourf(X,Y,T,20)    
    plt.colorbar()
    plt.quiver(X[::1,::1],Y[::1,::1],u[::1,::1],v[::1,::1],0) 
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Time Step %s' %(nt))


# Set the input parameters of the system
nx = 41
ny = 41
nit= 500
c = 1
dx = 1./(nx-1)
dy = 1./(ny-1)
x = np.linspace(0,1,nx)
y = np.linspace(0,1,ny)
X,Y = np.meshgrid(x,y)
rho = 1
nu = 0.1
dt = .001
T0 = 0
T_high = 1
D = 0.1
g = 1
beta = 0.01
v0= np.zeros(nx)
for i in range(nx):
    v0[i]=np.sin(np.pi/10*i)  #The pertubation imposed on bottom surface
    
#PLOT AT DIFFERENT TIME STEP
    
fig = plt.figure(figsize=(15,18), dpi=300)

# Time step 500
nt=200
u = np.zeros((ny, nx))
v = np.zeros((ny, nx))
p = np.zeros((ny, nx)) 
T = np.ones((ny, nx))*T0
b = np.zeros((ny, nx))
u, v, p, T = cavityFlow(nt, u, v, v0, dt, dx, dy, p, rho, nu, D, T, T0, T_high, g, beta)
ax = fig.add_subplot(326)
makegraph()