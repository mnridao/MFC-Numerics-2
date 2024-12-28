# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 15:46:41 2024

@author: mn1215
"""

class Parameters:
    """ 
    Class that manages and stores the physical parameters of the problem.
    """
    
    def __init__(self, nu, U):        
        """ 
        Initialise the physical parameters of the problem.
        
        @param nu: float representing diffusion coefficient
        @param U: callable object that returns the velocity field. I did it 
        this way because I wanted to put a swirly velocity field on the grid.
        """
        
        self.nu = nu
        self.U = U