# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 12:10:43 2024

@author: mn1215
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy import sparse as sp
from scipy import integrate

import helpers

# TODO: Refactor to avoid so much repetition and lambdas.

def calculateElementMassMatrix(x):
    """ 
    """
    
    # Reference shape function derivatives (array).
    dNdxi = helpers.referenceShapeFunctionDerivatives()
    
    # Calculate the Jacobian, its determinant and its inverse.
    J = helpers.jacobian(x, dNdxi)
    detJ = np.abs(np.linalg.det(J))
    
    # Gauss quadrature step.
    N = helpers.referenceShapeFunctions
    psi = lambda xi: detJ*np.outer(N(xi), N(xi))
    
    # Calculate me using Gauss quadrature and return result.
    return helpers.gaussQuadrature(psi)

def calculateElementStiffnessMatrix(x):
    """ 
    For diffusion.
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
    
def calculateElementStiffnessMatrix2(x, u):
    """ 
    For advection.
    
    u is [2x1]
    """
    
    # Reference shape function derivatives (array).
    dNdxi = helpers.referenceShapeFunctionDerivatives()
    
    # Calculate the Jacobian, its determinant and its inverse.
    J = helpers.jacobian(x, dNdxi)
    detJ = np.abs(np.linalg.det(J))
    Jinv = np.linalg.inv(J)
    
    # Lambda function for now.
    N = helpers.referenceShapeFunctions
    psi = lambda xi: -detJ*(dNdxi @ Jinv @ u @ N(xi).reshape(1, -1))
    
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

#%%
if __name__ == "__main__":
        
    # Generate the 2D grid.
    nx = 25
    nodes, IEN, ID, boundaries = helpers.generateGrid2D([0, 1], nx)
    
    # Construct the location matrix.
    LM = helpers.constructLM(IEN, ID)
        
    # Initialise global element and force matrices.
    numEqs = np.max(ID)+1    
    M = sp.lil_matrix((numEqs, numEqs))
    K = sp.lil_matrix((numEqs, numEqs))
    F = np.zeros((numEqs,))
    
    # Constant source term. 
    S = lambda x: 0.
    # S = lambda x: 1.
    # S = lambda x: x[0]
    # S = lambda x: 2*x[0]*(x[0]-2)*(3*x[1]**2-3*x[1]+0.5)+x[1]**2*(x[1]-1)**2
    
    # def S(x):
    #     """ 
    #     Gaussian bump."""
        
    #     # Define the mean and covariance for the Gaussian bump
    #     mean = [0.5, 0.5]  # Center of the Gaussian bump
    #     cov = [[0.0001, 0], [0, 0.0001]]  # Covariance matrix (controls spread and shape)

    #     # Compute the Gaussian bump value
    #     diff = np.array(x) - mean
    #     exponent = -0.5 * np.dot(diff.T, np.linalg.solve(cov, diff))
    #     normalization = 1 / (2 * np.pi * np.sqrt(np.linalg.det(cov)))
    #     return normalization * np.exp(exponent)
    
    # Physical parameters.
    u = np.array([1, 1]).reshape(-1, 1)
    nu = 0.
    
    # Loop over elements
    nElements = IEN.shape[0]
    for e in range(nElements):
        
        # Global element node coordinates.
        x = nodes[IEN[e,:],:]
        
        # Calculate element mass matrix.
        me = calculateElementMassMatrix(x)
        
        # Calculate element stiffness (currently just diffusion).        
        ke1 = calculateElementStiffnessMatrix(x)
        
        ke2 = calculateElementStiffnessMatrix2(x, u)
        
        ke = nu*ke1 + ke2
        # ke = ke1
        # ke = ke2
        
        # Calculate element force vector.        
        fe = calculateElementForceVector(x, S)
                
        # Populate the global matrices.
        for a in range(3):
            A = LM[a, e]
            for b in range(3):
                B = LM[b, e]
                if (A >= 0) and (B >= 0):
                    M[A, B] += me[a, b]
                    K[A, B] += ke[a, b]
            if (A >= 0):
                F[A] += fe[a]
    
   # %%
   
    from scipy.stats import multivariate_normal
    
    # Grid stuff
    xbounds = [0, 1]
    ybounds = xbounds
    nnodes = nx + 1
    
    x = np.linspace(xbounds[0], xbounds[1], nnodes)
    y = np.linspace(ybounds[0], ybounds[1], nnodes)
    X, Y = np.meshgrid(x, y)
    
    # Initial condition (would need to map to thing).
    intPsi0 = np.zeros(F.shape[0])  # :(
    
    # Define the mean and covariance for the Gaussian bump
    mean = [0.5, 0.5]  # Center of the Gaussian bump
    cov = [[0.0001, 0], [0, 0.0001]]  # Covariance matrix (controls spread and shape)

    # Create the Gaussian bump
    rv = multivariate_normal(mean, cov)
    Z = rv.pdf(np.dstack((X, Y)))  # Evaluate the PDF on the grid   
    Z /= Z.max()
    
    plt.contourf(X, Y, Z)
    #%%
    psi0 = Z.reshape(-1)
    
    # Joey code.
    K = sp.csr_matrix(K)
    M_inv = sp.linalg.inv(M)
    M_inv = sp.csr_matrix(M_inv)
    
    def rhs(t, psi):
        dpsidt = np.zeros_like(psi)
        dpsidt[ID >= 0] = M_inv @ (F - K @ psi[ID >= 0])
        return dpsidt
    
    #%
    plotEveryN = 1
        
    # Iterate in time.
    dt = 0.01
    endtime = 10
    
    soln = integrate.solve_ivp(rhs, [0, endtime], psi0, method='RK45',
                           max_step= dt)
        
    for i in range(1, soln.y.shape[1]):
        
        # if plotEveryN % i == 0:
            
        plt.tripcolor(nodes[:, 0], nodes[:, 1], soln.y[:, i], triangles=IEN)
        plt.title('Numeric')

        plt.colorbar()
        plt.tight_layout()

        plt.show()
            
    
    # t = 0
    # count = 0
    # while t < endtime:
    #     t += dt
    #     count += 1
        

        
    #     break
    #     # # RK2 step 1
    #     # dpsidt = np.linalg.solve(M, F - K@intPsi0)
    #     # intPsi = intPsi0 + dt*np.linalg.solve(M, F - K@intPsi0)
        
    #     # # RK2 step 2
    #     # dpsidt = np.linalg.solve(M, F - K @ intPsi)
    #     # intPsi2 = (intPsi + intPsi0 + dt * dpsidt) / 2
        
    #     # # Update old value.
    #     # intPsi0 = intPsi2.copy()
                
    #     # # Calculate full thing for plotting.
    #     # psi = np.zeros(nodes.shape[0])
    #     # for n in range(nodes.shape[0]):
    #     #     if ID[n] >= 0: # Otherwise psi should be zero.
    #     #         psi[n] = intPsi[ID[n]]
        
    #     # if plotEveryN % count == 0:
    #     plt.tripcolor(nodes[:, 0], nodes[:, 1], psi, triangles=IEN)
    #     plt.title('Numeric')

    #     plt.colorbar()
    #     plt.tight_layout()

    #     plt.show()
        
    # Add boundaries back in (disgusting).
    
    
    #%% Solve
    
    # Add a bump to initial condition.
    
    
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