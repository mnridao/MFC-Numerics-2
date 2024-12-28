# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 12:10:43 2024

@author: mn1215
"""

import numpy as np
import matplotlib.pyplot as plt

import helpers

def calculateElementStiffnessMatrix(x):
    """ 
    """
    
    # Reference shape function derivatives (array).
    dNdxi = helpers.referenceShapeFunctionDerivatives()
    
    # Calculate the Jacobian, its determinant and its inverse.
    J = helpers.jacobian(x, dNdxi)
    detJ = np.abs(np.linalg.det(J))
    Jinv = np.linalg.inv(J)
    
    # Lambda function just so I don't have to pass args in to another func.
    psi = lambda xi: detJ*(dNdxi @ Jinv @ Jinv.T @ dNdxi.T)
    
    # Calculate ke using Gauss quadrature and return result.
    return helpers.gaussQuadrature(psi)
    
def calculateElementForceVector(x, S):
    """ 
    Some repetition here.
    """
    
    # Reference shape function derivatives (array).
    dNdxi = helpers.referenceShapeFunctionDerivatives()
    
    # Calculate the Jacobian, its determinant and its inverse.
    J = helpers.jacobian(x, dNdxi)
    detJ = np.abs(np.linalg.det(J))
    
    # Map local 
    N = helpers.referenceShapeFunctions
    psi = lambda xi: S(helpers.mapLocalToGlobal(xi, x))*N(xi)*detJ
    
    # Calculate fe using Gauss quadrature and return result.
    return helpers.gaussQuadrature(psi)
    
if __name__ == "__main__":
        
    # Generate the 2D grid.
    nx = 50
    nodes, IEN, ID, boundaries = helpers.generateGrid2D([0, 1], nx)
    
    # Construct the location matrix.
    LM = helpers.constructLM(IEN, ID)
        
    # Initialise global element and force matrices.
    numEqs = np.max(ID)+1
    K = np.zeros(shape=(numEqs, numEqs))
    F = np.zeros(shape=(numEqs, ))
    
    # Constant source term. 
    # S = lambda x: 1.
    # S = lambda x: x[0]
    S = lambda x: 2*x[0]*(x[0]-2)*(3*x[1]**2-3*x[1]+0.5)+x[1]**2*(x[1]-1)**2
    
    # Loop over elements
    nElements = IEN.shape[0]
    for e in range(nElements):
        
        # Global element node coordinates.
        x = nodes[IEN[e,:],:]
        
        # Calculate element stiffness.        
        ke = calculateElementStiffnessMatrix(x)
        
        # Calculate element force vector.        
        fe = calculateElementForceVector(x, S)
                
        # Populate the global matrices.
        for a in range(3):
            A = LM[a, e]
            for b in range(3):
                B = LM[b, e]
                if (A >= 0) and (B >= 0):
                    K[A, B] += ke[a, b]
            if (A >= 0):
                F[A] += fe[a]
    
    #%% Solve
    intPsi = np.linalg.solve(K, F)
    psi = np.zeros(nodes.shape[0])
    for n in range(nodes.shape[0]):
        if ID[n] >= 0: # Otherwise psi should be zero.
            psi[n] = intPsi[ID[n]]
            
    #%% Plot results.
    
    plt.tripcolor(nodes[:, 0], nodes[:, 1], psi, triangles=IEN)
    plt.title('Numeric')

    plt.colorbar()
    plt.tight_layout()

    plt.show()