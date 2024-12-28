# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 15:45:58 2024

@author: mn1215
"""

from abc import ABC, abstractmethod
import numpy as np
from scipy import sparse as sp
from scipy import integrate

from src.element import Element

class Solver(ABC):
    """ 
    Base class for the finite element solver. Inheritied by both the 
    static and temporal solvers. Responsible for constructing the global 
    matrices."""
    
    def __init__(self, mesh, params, S):
        """ 
        Initialise the base class solver.
        
        @param mesh: Mesh object containing the grid information.
        @param params: Parameters object containing the physical params.
        @S: callable function that returns the soure function evaluated at point.
        """
        
        self.mesh = mesh 
        self.params = params
        self.S = S
                
    def constructGlobalMatrices(self, massMatrixNeeded=False):
        """ 
        This method is adapted from Ian's original code. Assumes the matrices
        are time-independent.
                
        @param massMatrixNeeded: stupid flag"""
        
        # Initialise the global matrices.
        numEqs = np.max(self.mesh.ID)+1 
        self.M = (None if not massMatrixNeeded 
                  else sp.lil_matrix((numEqs, numEqs)))
        self.K = sp.lil_matrix((numEqs, numEqs))
        self.F = np.zeros((numEqs,))
        
        # Loop over elements in the mesh.
        nElements = self.mesh.IEN.shape[0]
        for e in range(nElements):
            
            # Global element node coordinates.
            x = self.mesh.nodes[self.mesh.IEN[e,:],:]
            
            # Construct representation for the current element.
            element = Element(x, self.S, self.params.U)
            
            # Calculate the element mass matrix.
            me = None if not massMatrixNeeded else element.calculateElementMass()
            
            # Calculate the element diffusion stiffness matrix.
            ke1 = (np.zeros_like(me) if self.params.nu==0 
                   else element.calculateElementDiffusionStiffness())
            
            # Calculate the element advection stiffness matrix.
            ke2 = (np.zeros_like(ke1) if self.params.U == None 
                   else element.calculateElementAdvectionStiffness())
            
            # Calculate total element stiffness matrix.
            ke = self.params.nu*ke1 - ke2
                        
            # Calculate element force vector.        
            fe = element.calculateElementForceVector()
            
            # Populate the global matrices.
            for a in range(3):
                A = self.mesh.LM[a, e]
                for b in range(3):
                    B = self.mesh.LM[b, e]
                    if (A >= 0) and (B >= 0):
                        if massMatrixNeeded: 
                            self.M[A, B] += me[a, b]
                        self.K[A, B] += ke[a, b]
                if (A >= 0):
                    self.F[A] += fe[a]
        
    @abstractmethod 
    def run(self):
        pass
    
class StaticSolver(Solver):
    """ 
    Class responsible for running the static solver for the finite element method."""
    
    def __init__(self, mesh, params, S):
        """ 
        """
        super().__init__(mesh, params, S)
        
        # Construct the global matrices for static case.
        self.constructGlobalMatrices(massMatrixNeeded=False)
            
    def run(self):
        """ 
        Run the static finite element solver."""
        
        K = sp.csr_matrix(self.K)
        psiInterior = sp.linalg.spsolve(K, self.F)
        
        psi = np.zeros(self.mesh.nodes.shape[0])
        for n in range(self.mesh.nodes.shape[0]):
            if self.mesh.ID[n] >= 0: # Otherwise psi should be zero (homo dirichlet).
                psi[n] = psiInterior[self.mesh.ID[n]]
        
        # Normalise solution.
        psi /= max(psi)
        
        return psi
        
class TemporalSolver(Solver):
    """ 
    Class responsible for running the temporal solver for the finite element method."""
    
    def __init__(self, mesh, params, S, dt, endtime):
        """ 
        """
        super().__init__(mesh, params, S)
        
        # Construct the global matrices for the temporal case.
        self.constructGlobalMatrices(massMatrixNeeded=True)
        
        # Solver parameters.
        self.dt = dt
        self.endtime = endtime
        
    def run(self):
        """
        Run the time evolved finite element solver."""
        
        # This assumes zero-everywhere initial condition, not very general.
        psi = np.zeros(self.mesh.nodes.shape[0])
        
        # Convert to compressed sparse row matrix to work with sparse.
        K = sp.csr_matrix(self.K)
        M = sp.csc_matrix(self.M)
        M_inv = sp.linalg.inv(M)
        
        # Run the solution using RK45.
        def rhs(t, psi):        
            dpsidt = np.zeros_like(psi)
            dpsidt[self.mesh.ID >= 0] = M_inv @ (self.F - K @ psi[self.mesh.ID >= 0])
            return dpsidt
        res = integrate.solve_ivp(rhs, [0, self.endtime], psi, method='RK45', 
                                  max_step=self.dt)
                
        # Doesn't check for divide by zero.        
        return res.y[:, 1:]/res.y[:, 1:].max(0)