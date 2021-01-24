# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 13:24:17 2020

@author: mclea
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

# Domain
L = 1.
n = 101
h = L/(n-1)
x = y = np.linspace(0,L,n)
X,Y = np.meshgrid(x,y)

# Source term
f = np.zeros((n,n))
for i in range(n):
    for j in range(n):
        f[i,j] = 2*x[i]*(x[i]-1)+2*y[j]*(y[j]-1)

# Analytical solution
u_a = np.zeros((n,n))
for i in range(n):
    for j in range(n):
        u_a[i,j] = x[i]*y[j]*(x[i]-1)*(y[j]-1)
        
# Initial guess
u0 = np.ones((n,n))/20
u0[0,:] = u0[-1,:] = u0[:,0] = u0[:,-1] = 0

# Plot analytical solution & initial guess
fig = plt.figure(figsize=(16,6), dpi=50)
ax = fig.add_subplot(121, projection='3d')
ax.plot_surface(X,Y,u_a,rstride=5,cstride=5), plt.title('Analytical solution')
ax.set_xlabel('x'), ax.set_ylabel('y'), ax.set_zlabel('u'), ax.set_zlim3d(bottom=0,top=0.07)
ax = fig.add_subplot(122, projection='3d')
ax.plot_surface(X,Y,u0,rstride=5,cstride=5), plt.title('Initial guess')
ax.set_xlabel('x'), ax.set_ylabel('y'), ax.set_zlabel('u'), ax.set_zlim3d(bottom=0,top=0.07)
plt.show()


#Jacobi Iteration Method
def error(u):
    error = u-u_a
    err = (abs(error)).max()
        
    return err

def plt2d(err,it,title):
    fig.add_subplot(111)
    plt.plot(np.arange(1,it+1),err)
    plt.title(title), plt.xlabel('iterations'), plt.ylabel('maximum error')

def jacobi(u,f,h,max_err,max_it):
    t = time.time()
    
    u_n = u.copy()
    conv = []
    it = 0
    while True:
        it = it+1
        u_n[1:-1,1:-1] = 0.25*(u_n[2:,1:-1] + u_n[:-2,1:-1] + u_n[1:-1,2:] + u_n[1:-1,:-2] - f[1:-1,1:-1]*h*h)
        
        err = error(u_n)
        conv = np.concatenate((conv,[err]))
        
        if err < max_err:
            break
            
        if it > max_it:
            break
    
    t = time.time() - t
    
    # print 'Computation time = ' + ('%.5f' %t) + 's'
    # print 'Iterations =', it
    # print 'Maximum error = ' + ('%.4f' %err)
    # #plt3d(u_n, 'Jacobi iteration method')
    # plt2d(conv,it,'Jacobi iteration method')
    
    return u_n, it, conv, t

u_j, it_j, conv_j, t_j = jacobi(u0,f,h,0.001,10000)

fig = plt.figure(figsize=(16,6), dpi=50)
ax = fig.add_subplot(121, projection='3d')
ax.plot_surface(X, Y, u_a, rstride=5, cstride=5)
plt.title('Analytical solution')
ax.set_xlabel('x'), ax.set_ylabel('y'), ax.set_zlabel('u'), ax.set_zlim3d(bottom=0,top=0.07)
ax = fig.add_subplot(122, projection='3d')
ax.plot_surface(X,Y,u_j,rstride=5,cstride=5), plt.title('Initial guess')
ax.set_xlabel('x'), ax.set_ylabel('y'), ax.set_zlabel('u'), ax.set_zlim3d(bottom=0,top=0.07)
plt.show()