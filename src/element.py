# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 19:31:30 2024

@author: mn1215
"""

import numpy as np

from src.shapeFunctions import LinearShapeFunction
import src.utils as utils
    
class Element:
    """
    Class responsible for performing calculations on a single element.
    """
    
    def __init__(self, x, S, U):
        """ 
        Initialises the element using linear shape functions.
        
        @param x: Array of (global) nodal coordinates for the element.
        @param S: Callable function that returns source function.
        @param U: Callable function that returns the velocity field over domain.
        """
        
        self.x = x
        self.S = S
        self.U = U
        
        # There is only the option for linear shape function.
        self.shapeFunction = LinearShapeFunction(x)
        
    def calculateElementMass(self):
        """ 
        Calculates the mass matrix for the element using Gauss quadrature.
        """
        return utils.gaussQuadrature(self.massIntegrand)
    
    def calculateElementAdvectionStiffness(self):
        """ 
        Calculates the advection term stiffness matrix for the element using 
        Gauss quadrature.
        """
        return utils.gaussQuadrature(self.advectionStiffnessIntegrand)
    
    def calculateElementDiffusionStiffness(self):
        """ 
        Calculates the diffusion term stiffness matrix for the element using 
        Gauss quadrature.
        """
        return utils.gaussQuadrature(self.diffusionStiffnessIntegrand)
    
    def calculateElementForceVector(self):
        """ 
        Calculates the force vector for the element using Gauss quadrature.
        """
        return utils.gaussQuadrature(self.forceIntegrand)
    
    ## INTEGRAND FUNCTIONS ##
    def massIntegrand(self, xi):
        """ 
        Returns the mass integrand for a certain xi, to be used with Gauss 
        quadrature. 
        
        @param xi: 
        """
        # Reference shape function at x.
        N = self.shapeFunction.referenceShapeFunctions(xi)
        
        return self.shapeFunction.detJ*np.outer(N, N)
    
    def advectionStiffnessIntegrand(self, xi):
        """ 
        Returns the advection stiffness integrand for a certain xi, to be 
        used with Gauss quadrature.
        
        I've given too much blood, sweat and tears to remove (but it was futile).
        
        @param xi: 
        """
        # Retrieve shape function data. 
        N = self.shapeFunction.referenceShapeFunctions(xi).reshape(1, -1)
        detJ = self.shapeFunction.detJ 
        Jinv = self.shapeFunction.invJ 
        dNdxi = self.shapeFunction.dNdxi
                
        # Compute the velocity at the global coordinates.
        globalCoords = self.shapeFunction.mapLocalToGlobal(xi, self.x)
        u = self.U(globalCoords).reshape(-1, 1)
        
        return detJ*(dNdxi @ Jinv @ u @ N).T
    
    def advectionStiffnessIntegrand_LOOPS(self, xi):
        """ 
        Returns the advection stiffness integrand for a certain xi, to be 
        used with Gauss quadrature.
        
        Never finished! 
        """
        # Retrieve shape function data. 
        N = self.shapeFunction.referenceShapeFunctions(xi).reshape(1, -1)
        detJ = self.shapeFunction.detJ 
        Jinv = self.shapeFunction.invJ 
        dNdxi = self.shapeFunction.dNdxi
                
        # Compute the velocity at the global coordinates.
        globalCoords = self.shapeFunction.mapLocalToGlobal(xi, self.x)
        u = self.U(globalCoords)
        
        ke = np.zeros(3, 3)
        
    
    def diffusionStiffnessIntegrand(self, xi):
        """ 
        Returns the diffusion stiffness integrand for a certain xi, to be 
        used with Gauss quadrature.
        
        @param xi: 
        """
        # Retrieve shape function data (makes easier to read).
        detJ = self.shapeFunction.detJ 
        Jinv = self.shapeFunction.invJ 
        dNdxi = self.shapeFunction.dNdxi
        
        return detJ*(dNdxi @ Jinv @ Jinv.T @ dNdxi.T)
    
    def forceIntegrand(self, xi):
        """ 
        Returns the force integrand for a certain xi, to be used with Gauss
        quadrature.
        
        @param xi: 
        """
        # Retrieve shape function data. 
        N = self.shapeFunction.referenceShapeFunctions(xi)
        detJ = self.shapeFunction.detJ
        
        # Compute the source function at the global coordinates.
        globalCoords = self.shapeFunction.mapLocalToGlobal(xi, self.x)
        source = self.S(globalCoords)
        
        return source*N*detJ