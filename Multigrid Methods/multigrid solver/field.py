# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 12:46:32 2021

@author: mclea
"""

import numpy as np

class new_field:
    """
    """
    
    def __init__(self, n, m, top=0, bottom=0, left=0, right=0, bc=1):
        self.n = n
        self.m = m
        self.x = np.zeros((self.n, self.m))
        self.top = top
        self.bottom = bottom
        self.left = right
        self.right = right
        self.bc = bc
        self.set_BC()
    
    def set_BC(self):
        
        if self.bc==0:
            # top
            self.x[0, :] = self.top
    
            # bottom
            self.x[-1, :] = self.bottom
        
            # left
            self.x[:, 0] = self.left
    
            # right
            self.x[:, -1] = self.right
