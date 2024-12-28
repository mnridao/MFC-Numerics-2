# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 15:47:12 2024

@author: mn1215
"""

import numpy as np

from src.shapeFunctions import LinearShapeFunction

## GAUSS QUADRATURE ##

def gaussQuadrature(psi):
    """ 
    Gauss quadrature function for 
    """
    xiGauss = np.array([[1/6, 1/6], [4/6, 1/6], [1/6, 4/6]])
    return 1/6*sum(psi(xi) for xi in xiGauss)

## SOURCE FUNCTIONS ##

def sotoFire(x):
    """
    Gaussian bump source representing pollution from the fire in Southampton.
    """
    sigma = 5000
    return np.exp(-1/(2*sigma**2)*((x[0]-442365)**2 + (x[1]-115483)**2))

## VELOCITY FIELDS ##

def windTowardsReading(x):
    return -10*np.array([0.49082847, 0.87125623])

def swirlyWhirlyWind(x):
    k = 1e5  # Scaling factor for velocity
    decay = 1e-5  # Decay factor for distance
    
    # Debug.
    k/=10
    decay*=2
    
    xc, yc = readingCoords
    r = np.sqrt((x[0] - xc)**2 + (x[1] - yc)**2)
    r = np.maximum(r, 1e-10)  # Prevent division by zero
    vx = -k * (x[1] - yc) * decay / r
    vy = k * (x[0] - xc) * decay / r
    return -10*np.array([vx, vy])

## USEFUL COORDINATES ##

readingCoords = np.array([473993, 171625])
sotoCoords = np.array([442365, 115483])

## MISC ##

def global2localCoords(xe, x):
    """
    Transforms global coordinates to local coordinates for a triangular element.
    All credit to Joey.
    """
    # Inverts the equation x = x0^e (1 - xi1 - xi2) + x1^e xi1 + x2^e xi2
    #                      y = y0^e (1 - xi1 - xi2) + y1^e xi1 + y2^e xi2
    # to solve for xi1, xi2.
    diffs = np.array([[xe[0,1]-xe[0,0], xe[0,2]-xe[0,0]],
                      [xe[1,1]-xe[1,0], xe[1,2]-xe[1,0]]])
    localcoords = np.linalg.solve(diffs, x-np.array([xe[0,0],xe[1,0]]))
    return localcoords

def areaTriangle(x1, x2, x3):
    """ 
    Find the area of the triangle given node coordinates.
    
    @param x1: array of first node.
    @param x2: array of second node.
    @param x3: array of third node.
    """
    
    area = abs(x1[0]*(x2[1]-x3[1]) + x2[0]*(x3[1]-x1[1]) + x3[0]*(x1[1]-x2[1]))
    return int(area)

def pointInElement(x, coords):
    """ 
    Check if a point is inside an element.
    
    @param x: array of nodes of the element.
    @param coords: array of the point coordinates.
    """
    
    # Calculate area of element (Nodes a - b - c).
    abcArea = areaTriangle(x[0, :], x[1, :], x[2, :])
    
    # Calculate area of element split by reading coords.
    pabArea = areaTriangle(coords, x[0, :], x[1, :])
    pbcArea = areaTriangle(coords, x[1, :], x[2, :])
    pacArea = areaTriangle(coords, x[0, :], x[2, :])
    
    alpha = pbcArea/abcArea 
    beta = pacArea/abcArea 
    gamma = pabArea/abcArea
    
    return alpha + beta + gamma < 1

def extractPollution(psi, mesh, coords):
    """ 
    Extracts the value of psi at the coordinate values. First the element that 
    coords are in is find, then the value of psi at that point is interpolated
    using the shape functions of that element.
    
    @param psi: array of pollution values 
    @param mesh: UKMesh object 
    @param coords: coordinate values at which to find pollution.
    """
    
    # Find which element coords are in.
    for e in range(mesh.IEN.shape[0]):
        
        # Nodal coordinates
        x = mesh.nodes[mesh.IEN[e,:],:]
                
        if pointInElement(x, coords):
            break
    
    # Construct the shape function for the element.    
    shapeFunction = LinearShapeFunction(x)
    
    xi = global2localCoords(x.T, coords)
    N = shapeFunction.referenceShapeFunctions(xi)
    
    return np.dot(psi[mesh.IEN[e, :]], N)