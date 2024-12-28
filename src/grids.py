# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 15:45:32 2024

@author: mn1215
"""
import numpy as np

class UKMesh:
    """ 
    Class responsible for storing the mesh information for the UK map."""
    
    def __init__(self, resolution, bc="SouthernBoundaryDirichlet"):
        """ 
        
        @param resolution: string with the required mesh resolution (must be one of ['1_25', '2_5', '5', '10', '20', '40'])
        @param bc: string describing the desired boundary condition.
        """
        
        # Define boundary condition type.
        self.bc = bc
        
        # Root file path - default is previous folder, should be more general.
        # self.root = "../"
        self.root = ""
        
        # Check if resolution is acceptable.
        acceptableRes = ['1_25', '2_5', '5', '10', '20', '40']
        if resolution not in acceptableRes:
            raise Exception(f"Resolution {resolution} not in {acceptableRes}.")
        
        self.initialise(resolution)
    
    def initialise(self, resolution):
        """
        Reads the grid data then generates the mesh.
        
        @param resolution: string with the required mesh resolution.
        """
        
        self.readGrid(resolution)
        self.constructID()
        self.constructLM()
    
    def readGrid(self, resolution):
        """ 
        
        @param resolution: """
        
        self.nodes = np.loadtxt(f'{self.root}las_grids/las_nodes_{resolution}k.txt')
        self.IEN = np.loadtxt(f'{self.root}las_grids/las_IEN_{resolution}k.txt', dtype=np.int64)
        self.boundary_nodes = np.loadtxt(f'{self.root}las_grids/las_bdry_{resolution}k.txt', dtype=np.int64)
    
    def constructID(self):
        """ 
        """
        
        if self.bc == "SouthernBoundaryDirichlet":
            dirichletBoundaries = np.where(self.nodes[self.boundary_nodes,1] <= 110000)[0]
        
        self.ID = np.zeros(len(self.nodes), dtype=np.int64)
        n_eq = 0
        for i in range(len(self.nodes[:, 1])):
            if i in dirichletBoundaries:
                self.ID[i] = -1
            else:
                self.ID[i] = n_eq
                n_eq += 1
    
    def constructLM(self):
        """ 
        """
        self.LM = np.zeros_like(self.IEN.T)
        for e in range(self.IEN.shape[0]):
            for a in range(self.IEN.shape[1]):
                self.LM[a, e] = self.ID[self.IEN[e, a]]