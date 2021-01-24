# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 13:20:01 2021

@author: mclea
"""

import field as f
import possion_matrix
import MG
import matplotlib.pyplot as plt

k = 8
n = 2**k+2
m = 2**k+2
x = f.new_field(n, m, top=10, bottom=-10, left=0, right=0, bc=0)
b = f.new_field(n, m, top=0, bottom=0, left=0, right=0)

pm = possion_matrix.possion_matrix(x)
x = MG.MGV(x, b, pm)
plt.imshow(x.x)
# A, LaddU, invD = possion_matrix.create_A(field.n, field.n)
# Rj = possion_matrix.calc_RJ(field.n, field.n)
# MG.jacobi_update(x, b, pm.A[0], pm.Rj[0], pm.invD[0], nsteps=2000, max_err=1e-3)
# plt.imshow(x.x)