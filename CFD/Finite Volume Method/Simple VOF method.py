# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 14:06:48 2020

@author: mclea
"""
import numpy as np
import matplotlib.pyplot as plt

def makegraph(x,y,u,v,nx,ny,dx,dy,r,steps):
    fig = plt.figure(figsize=(6,6), dpi=300)
    uu=0.5*(u[0:nx,1:ny+1]+u[0:nx,0:ny])
    vv=0.5*(v[1:nx+1,0:ny]+v[0:nx,0:ny])
    yy,xx=np.mgrid[0:(nx-1)*dx:nx*1j,0:(ny-1)*dx:ny*1j]
    plt.contourf(x,y,r.T,5)
    plt.colorbar
    plt.quiver(xx,yy,uu.T,vv.T)
    plt.title('Time Step %s' %(steps))

# Domain and physical variables
Lx = 1.0
Ly = 1.0
gx = 0
gy = -100
rho1 = 1.0
rho2 = 2.0
m0 = 0.01
rro = rho1
unorth = 0
usouth = 0
veast = 0
vwest = 0
time = 0.0

# Initial drop size and location
rad = 0.15
xc = 0.5
yc = 0.5


# Numerical variables
nx = 32
ny = 32
dt = 0.005
nstep = 300
maxit = 200
maxError = 0.001
beta = 1.2

# Zero arrays
u = np.zeros((nx+1, ny+2))
v = np.zeros((nx+2, ny+1))
p = np.zeros((nx+2, ny+2))
ut = np.zeros((nx+1, ny+2))
vt = np.zeros((nx+2, ny+1))
tmp1 = np.zeros((nx+2, ny+2))
uu = np.zeros((nx+1, ny+1))
vv = np.zeros((nx+1, ny+1))
tmp2 = np.zeros((nx+2, ny+2))

# Create the grid
dx = Lx/nx
dy = Ly/ny
x = np.zeros(nx+2)
y = np.zeros(ny+2)

for i in range(len(x)):
    x[i] = dx*(i-0.5)

for j in range(len(y)):
    y[j] = dy*(j-0.5)

XX, YY = np.meshgrid(x, y)

in_circle = (XX-xc)**2 + (YY-yc)**2 < rad**2

r = np.zeros((len(x), len(y))) + rho1
r[in_circle] = rho2


# =============================================================================
# Time loop
# =============================================================================
for i_s in range(nstep):
    # Tangental velocity at the boundaries
    u[:, 0] = 2*usouth - u[:, 1] # left wall
    u[:, -1] = 2*unorth - u[:, -2] # right wall
    v[0, :] = 2*vwest - v[1, :] # Top wall
    v[-1, :] = 2*veast - v[-2, :] # Bottom wall
    
    ut[1:-1,1:-1]=u[1:-1,1:-1]+dt*(-0.25*(((u[2:,1:-1]+u[1:-1,1:-1])**2-(u[1:-1,1:-1]+u[0:-2,1:-1])**2)/dx+\
                                          ((u[1:-1,2:]+u[1:-1,1:-1])*(v[2:-1,1:]+v[1:-2,1:])-\
                                           (u[1:-1,1:-1]+u[1:-1,0:-2])*(v[2:-1,0:-1]+v[1:-2,0:-1]))/dy)+\
                                           m0/(0.5*(r[2:-1,1:-1]+r[1:-2,1:-1]))*\
                                        ((u[2:,1:-1]-2*u[1:-1,1:-1]+u[0:-2,1:-1])/(dx**2)+\
                                        (u[1:-1,2:]-2*u[1:-1,1:-1]+u[1:-1,0:-2])/(dy**2))+gx)

        #Temporary v-velocity
    vt[1:-1,1:-1]=v[1:-1,1:-1]+dt*(-0.25*(((u[1:,2:-1]+u[1:,1:-2])*(v[2:,1:-1]+v[1:-1,1:-1])-\
                                               (u[0:-1,2:-1]+u[0:-1,1:-2])*(v[1:-1,1:-1]+v[0:-2,1:-1]))/dx+\
                                              ((v[1:-1,2:]+v[1:-1,1:-1])**2-(v[1:-1,1:-1]+v[1:-1,0:-2])**2)/dy)+\
                                       m0/(0.5*(r[1:-1,2:-1]+r[1:-1,1:-2]))*\
                                       ((v[2:,1:-1]-2*v[1:-1,1:-1]+v[0:-2,1:-1])/(dx**2)+\
                                        (v[1:-1,2:]-2*v[1:-1,1:-1]+v[1:-1,0:-2])/(dy**2))+gy)
    
    # Compute source term and the coefficient for p(i,j)
    rt = r.copy()
    lrg = 1000
    rt[:, 0] = lrg
    rt[:, -1] = lrg
    rt[0, :] = lrg
    rt[-1, :] = lrg

    tmp1[1:-1,1:-1]=(0.5/dt)*((ut[1:,1:-1]-ut[0:-1,1:-1])/dx+(vt[1:-1,1:]-vt[1:-1,0:-1])/dy)
    tmp2[1:-1,1:-1]=1.0/((1./dx)*(1./(dx*(rt[2:,1:-1]+rt[1:-1,1:-1]))+\
                                  1./(dx*(rt[0:-2,1:-1]+rt[1:-1,1:-1])))+\
                         (1./dy)*(1./(dy*(rt[1:-1,2:]+rt[1:-1,1:-1]))+\
                                  1./(dy*(rt[1:-1,0:-2]+rt[1:-1,1:-1]))))
    
    # Poisson solver for p:
    iter=0
    while True:
        pn=p.copy()
        iter=iter+1
        for i in range(1,nx+1):
            for j in range(1,ny+1):
                p[i,j]=(1.0-beta)*p[i,j]+beta*tmp2[i,j]*(\
                    (1./dx)*( p[i+1,j]/(dx*(rt[i+1,j]+rt[i,j]))+\
                    p[i-1,j]/(dx*(rt[i-1,j]+rt[i,j])))+\
                    (1./dy)*( p[i,j+1]/(dy*(rt[i,j+1]+rt[i,j]))+\
                    p[i,j-1]/(dy*(rt[i,j-1]+rt[i,j])))-tmp1[i,j])

        if np.abs(pn-p).max()<maxError:
            break
        if iter>maxit:
            break
    
    #Calculate u-velocity:
    u[1:-1,1:-1]=ut[1:-1,1:-1]-dt*(2.0/dx)*(p[2:-1,1:-1]-p[1:-2,1:-1])/(r[2:-1,1:-1]+r[1:-2,1:-1])


    #Calculate v-velocity:
    v[1:-1,1:-1]=vt[1:-1,1:-1]-dt*(2.0/dy)*(p[1:-1,2:-1]-p[1:-1,1:-2])/(r[1:-1,2:-1]+r[1:-1,1:-2])
    
    ro = r.copy()
    for i in range(1, nx):
        for j in range(1, ny):
            r[i,j] = (ro[i,j] -
                     ((0.5*dt/dx)*(u[i,j]*(ro[i+1,j]+ro[i,j]) -
                                   u[i-1,j]*(ro[i-1,j]+ro[i,j]))) -
                     ((0.5*dt/dy)*(v[i,j]*(ro[i,j+1]+ro[i,j]) -
                                   v[i,j-1]*(ro[i,j-1]+ro[i,j]))) +
                     ((m0*dt/(dx*dx))*(ro[i+1,j]-2.0*ro[i,j]+ro[i-1,j])) +
                     ((m0*dy/(dy*dy))*(ro[i,j+1]-2.0*ro[i,j]+ro[i,j-1])))
    
    makegraph(x, y, u, v, nx, ny, dx, dy, r, i_s)
                      