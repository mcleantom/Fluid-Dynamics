# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 10:48:41 2021

@author: mclea
"""
import numpy as np
from scipy.sparse import csr_matrix

class possion_matrix:
    """
    """
    
    def __init__(self, field):
        self.n = field.n
        self.levels = 0
        self.A, self.Rj, self.invD, self.layers = levels(self.n, self.n)


def adjacency_matrix(rows, cols):
    """
    Creates the adjacency matrix for an n by m shaped grid
    """
    n = rows*cols
    M = np.zeros((n,n))
    for r in range(rows):
        for c in range(cols):
            i = r*cols + c
            # Two inner diagonals
            if c > 0: M[i-1,i] = M[i,i-1] = 1
            # Two outer diagonals
            if r > 0: M[i-cols,i] = M[i,i-cols] = 1
    return M


def create_differences_matrix(rows, cols):
    """
    Creates the central differences matrix A for an n by m shaped grid
    """
    n = rows*cols
    M = np.zeros((n,n))
    for r in range(rows):
        for c in range(cols):
            i = r*cols + c
            # Two inner diagonals
            if c > 0: M[i-1,i] = M[i,i-1] = -1
            # Two outer diagonals
            if r > 0: M[i-cols,i] = M[i,i-cols] = -1
    np.fill_diagonal(M, 4)
    return M


def create_A(n,m):
    """
    Creates all the components required for the jacobian update function
    for an n by m shaped grid
    """
    LaddU = adjacency_matrix(n,m)
    A = create_differences_matrix(n,m)
    invD = np.zeros((n*m, n*m))
    np.fill_diagonal(invD, 1/4)
    return A, LaddU, invD


def calc_RJ(rows, cols):
    """
    Calculates the jacobian update matrix Rj for an n by m shaped grid
    """
    n = int(rows*cols)
    M = np.zeros((n,n))
    for r in range(rows):
        for c in range(cols):
            i = r*cols + c
            # Two inner diagonals
            if c > 0: M[i-1,i] = M[i,i-1] = 0.25
            # Two outer diagonals
            if r > 0: M[i-cols,i] = M[i,i-cols] = 0.25

    return M


def create_jacobi_update_arrays(n,m):
    A, LaddU, invD = create_A(n,m)
    Rj = calc_RJ(n,m)
    A = csr_matrix(A)
    LaddU = csr_matrix(LaddU)
    invD = csr_matrix(invD)
    Rj = csr_matrix(Rj)
    return A, Rj, invD


def levels(n,m):
    """
    """
    shape = []
    A = []
    Rj = []
    invD = []
    shape.append([n,m])
    next_level = True
    level = 0
    while next_level:
        A_new, Rj_new, invD_new = create_jacobi_update_arrays(n-2,m-2)
        shape.append([n,m])
        A.append(A_new)
        Rj.append(Rj_new)
        invD.append(invD_new)                        
        level += 1
        n, m = restrict(n,m)
        
        if n<6 or m<6:
            next_level = False
    return A, Rj, invD, level
    

def restrict(n,m):
    new_n = int((n-2)/2+2)
    new_m = int((m-2)/2+2)
    return new_n, new_m