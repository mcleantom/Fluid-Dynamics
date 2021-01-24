# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 09:02:05 2020

@author: mclea
"""

from __future__ import division
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np

from scipy import linalg

def implicitDiffusion(Nt, Nx, L, T, D):
    """
    """
    dt = L/Nt
    dx = T/Nx
    alpha = D*dt/(dx*dx)
    
    x = np.linspace(0, L, Nx)
    t = np.linspace(0, T, Nt)
    u = np.zeros((Nx, Nt))
    
    # Initial condition
    u[:,0] = np.sin(np.pi*x)
    
    u[0,:] = 0
    u[-1,:] = 0
    
    aa = -alpha*np.ones(Nx-3)
    bb = (1+2*alpha)*np.ones(Nx-2)
    cc = -alpha*np.ones(Nx-3)
    M = np.zeros((len(bb), len(bb)))
    M = np.diag(aa, -1) + np.diag(bb, 0) + np.diag(cc,1)
    
    for k in range(1, Nt):
        u[1:-1, k] = linalg.solve(M, u[1:-1, k-1])
    
    return u, x, t, alpha

fig = plt.figure(figsize=(12,6))
plt.rcParams['font.size'] = 15

ax = fig.add_subplot(121, projection='3d')
ui, xi, ti, alphai = implicitDiffusion(Nt=2500, Nx=50, L=1., T=1., D=2.5)
Ti, Xi = np.meshgrid(ti,xi)
N = ui/ui.max()
ax.plot_surface(Ti, Xi, ui, linewidth=0, facecolors=cm.jet(N), rstride=1, cstride=50)
ax.set_xlabel('Time $t$')
ax.set_ylabel('Distance $x$')
ax.set_zlabel('Concentration $u$')
ax.set_title('$\\alpha = 0.25$')
plt.tight_layout()

ax = fig.add_subplot(122, projection='3d')
ui1, xi1, ti1, alphai1 = implicitDiffusion(Nt = 2500, Nx = 50, L= 1., T = 1, D =0.25)
Ti1, Xi1 = np.meshgrid(ti1,xi1)
N = ui1/ui1.max()
ax.plot_surface(Ti1, Xi1, ui1, linewidth=0, facecolors=cm.jet(N), rstride=1, cstride=50 )
ax.set_xlabel('Time $t$')
ax.set_ylabel('Distance $x8$')
ax.set_zlabel('Concentration $u$')
ax.set_title('$\\alpha = 0.505$')
plt.tight_layout()