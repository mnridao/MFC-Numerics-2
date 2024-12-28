# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 10:54:27 2024

@author: mn1215
"""

import numpy as np

### LINEAR SHAPE FUNCTIONS ###

def referenceShapeFunctions(xi):
    """ 
    """
    return np.array([1. - xi[0] - xi[1], xi[0], xi[1]])

def referenceShapeFunctionDerivatives():
    """ 
    """
    return np.array([[-1., -1.], [1., 0.], [0., 1.]])

### SETTING UP GRID ###

def generateGrid2D(xbounds, nx, ybounds=None, ny=None):
    """
    Modified from Ian
    """
    
    # Default grid is square.
    ybounds = ybounds if ybounds else xbounds 
    ny = ny if ny else nx
    
    # Number of nodes.
    nnodes = nx+1 
    
    # Generate the grid points.
    x = np.linspace(xbounds[0], xbounds[1], nnodes)
    y = np.linspace(ybounds[0], ybounds[1], nnodes)
    X, Y = np.meshgrid(x, y)
    
    nodes = np.zeros((nnodes**2, 2))
    nodes[:, 0] = X.ravel() 
    nodes[:, 1] = Y.ravel()
    
    # Construct ID (links global node number to global eqn number).
    ID, boundaries = constructID(nodes)
    
    # Construct IEN (links the element + local node no. to global node no.).
    IEN = constructIEN(nodes, nx)
    
    return nodes, IEN, ID, boundaries 
    
def constructID(nodes):
    """ 
    """
    
    ID = np.zeros(len(nodes), dtype=np.int64)
    boundaries = dict()  # Will hold the boundary values
    n_eq = 0
    for nID in range(len(nodes)):
        
        # TODO: Add more boundary condition options.
        
        # Left-hand side is dirichlet I think.
        if np.allclose(nodes[nID, 0], 0):
            ID[nID] = -1
            boundaries[nID] = 0  # Dirichlet BC
        else:
            ID[nID] = n_eq
            n_eq += 1
            if ( (np.allclose(nodes[nID, 1], 0)) or 
                 (np.allclose(nodes[nID, 0], 1)) or 
                 (np.allclose(nodes[nID, 1], 1)) ):
                boundaries[nID] = 0 # Neumann BC
    return ID, boundaries

def constructIEN(nodes, nx):
    """ 
    """
    nnodes = nx + 1
    
    IEN = np.zeros((2*nx**2, 3), dtype=np.int64)
    for i in range(nx):
        for j in range(nx):
        
            # Bottom left triangle.
            IEN[2*i+2*j*nx  , :] = (i+j*nnodes, 
                                    i+1+j*nnodes, 
                                    i+(j+1)*nnodes)
            
            # Top right triangle.
            IEN[2*i+1+2*j*nx, :] = (i+1+j*nnodes, 
                                    i+1+(j+1)*nnodes, 
                                    i+(j+1)*nnodes)
    return IEN

def constructLM(IEN, ID):
    """ 
    """
    LM = np.zeros_like(IEN.T)
    for e in range(IEN.shape[0]):
        for a in range(IEN.shape[1]):
            LM[a, e] = ID[IEN[e, a]]
    return LM

### GAUSS QUADRATURE STUFF ###

def gaussQuadrature(psi):
    """ 
    """
    xiGauss = np.array([[1/6, 1/6], [4/6, 1/6], [1/6, 4/6]])
    return 1/6*sum(psi(xi) for xi in xiGauss)

### MISC ###

def jacobian(x, dNdxi):
    """ 
    """
    return x.T @ dNdxi

def mapLocalToGlobal(xi, x):
    """ 
    """
    return x.T @ referenceShapeFunctions(xi)
