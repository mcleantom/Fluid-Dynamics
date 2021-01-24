# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 12:46:32 2021

@author: mclea
"""

import numpy as np

class field:
    """
    """
    
    def __init__(self, k):
        self.n = 2**k+2
        self.x = np.zeros((self.n, self.n))
    
    def set_BC(self):
        # bottom
        self.x[0, 0:(int(self.n/2))] = np.linspace(13, 5, int((self.n/2)))
        self.x[0, int(self.n/2):int(3*self.n/4)] = 5
        self.x[0, int(3*self.n/4):-1] = np.linspace(5, 13, int(self.n/4))
    
        # top
        self.x[-1, :] = 21
    
        # left
        self.x[0:int(3*self.n/8), 0] = np.linspace(13, 40, int(3*self.n/8))
        self.x[int(self.n/2):, 0] = np.linspace(40, 21, int(self.n/2))
    
        # right
        self.x[0:int(self.n/2), -1] = np.linspace(13, 40, int(self.n/2))
        self.x[int(5*self.n/8):-1, -1] = np.linspace(40, 21, int(3*self.n/8))
    
        # heaters
        self.x[int(3*self.n/8):int(self.n/2), 0:int(self.n/8+1)] = 40
        self.x[int(self.n/2):int(5*self.n/8), int(7*self.n/8):(self.n+1)] = 40

f = field(9)
f.set_BC()