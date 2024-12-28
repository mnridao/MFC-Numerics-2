# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 17:13:13 2024

@author: mn1215
"""

import numpy as np

class LinearShapeFunction:
    """ 
    Class responsible for defining and managing the properties and computations 
    of linear shape functions."""
    
    # # Global attribute I've never done this before? Is this illegal?
    # dNdxi = np.array([[-1., -1.], [1., 0.], [0., 1.]])
    
    def __init__(self, x):
        """
        Initialise the shape function with nodal coordinates.
        
        @param x: Array of nodal coordinates defining the finite element.
        """
        # Calculate the Jacobian, its determinant and its inverse.
        self.calculateShapeFunctionProperties(x)
    
    def calculateShapeFunctionProperties(self, x):
        """ 
        Calculate the shape function derivatives, and the Jacobian, its 
        determinent and its inverse.
        
        @param x: Array of nodal coordinates defining the finite element.
        """
        
        # Shape function derivatives in 2D.
        self.dNdxi = np.array([[-1., -1.], [1., 0.], [0., 1.]])
        
        self.J = x.T @ self.dNdxi
        # TODO: Change to sparse.
        self.detJ = np.linalg.det(self.J)
        self.invJ = np.linalg.inv(self.J)
    
    def referenceShapeFunctions(self, xi):
        """
        Evaluate the shape function values in the reference (local) 
        coordinate system.
        
        @param xi: Array of reference coordinates. 
        """
        return np.array([1. - xi[0] - xi[1], xi[0], xi[1]])
    
    def globalShapeFunctionDerivatives(self):
        """ 
        Compute the derivatives of the shape functions with respect to global 
        coordinates.
        """
        return self.dNdxi @ self.Jinv
    
    def mapLocalToGlobal(self, xi, x):
        """ 
        Map a point from local (reference element) coordinates to global 
        coordinates.
        
        @param xi: Array of reference coordinates. 
        @param x : Array of global coordinates.
        """
        return x.T @ self.referenceShapeFunctions(xi)