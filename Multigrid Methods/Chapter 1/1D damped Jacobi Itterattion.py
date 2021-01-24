# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 14:21:07 2020

@author: mclea
"""

import numpy as np
import matplotlib.pyplot as plt

def fill_offset_diagonal(u, k, val):
    u[np.eye(len(u), k=k, dtype='bool')] = val
    return u

sigma = 0

# Set up a grid
# Number of sub-intervals
n = 100
n += 1
h = 1/n
L = 1
x = np.linspace(0, L, n)

# The slope
f = np.ones(n)
f = np.vstack(f)

# Approximation to the exact solution
v = np.vstack(np.zeros(n))

A = np.zeros((n,n))
np.fill_diagonal(A, (2+sigma*h**2))
A = fill_offset_diagonal(A, -1, -1)
A = fill_offset_diagonal(A, 1, -1)
# A = (1/h**2)*A

f = (h**2)*f

# The diagonal, left of the diagonal and right of the diagonal
D = np.zeros((n,n))
D = fill_offset_diagonal(D, 0, np.diag(A, 0))
L = -1*np.tril(A, -1)
U = -1*np.triu(A, 1)

w = 0.9 

invD = np.linalg.inv(D)
Rj = np.dot(invD, (L+U))

I = np.identity(n)
Rw = (1-w)*I + w*Rj

nsteps = 100

for i in range(nsteps):
    v = np.dot(Rw,v) + w*np.dot(invD,f)

plt.plot(x, v)

max_error = max(f-np.dot(A,v))[0]
print("The maximum error is " + str(max_error))