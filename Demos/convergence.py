# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 22:14:31 2024

@author: mn1215
"""

import numpy as np

from src.grids import UKMesh
from src.parameters import Parameters
from src.solver import StaticSolver
import src.utils as utils

import Demos.plotters as plt

def runStaticConvergenceAnalysis():
    """ 
    Runs an experimental convergence analysis for the static finite element 
    solver for resolutions between 40km to 2.5km. The finest resolution (1.25km) 
    is used as the reference solution."""
    
    debug = True
    
    # Resolutions to iterate over.
    resolutions = ["40", "20", "10", "5", "2_5", "1_25"]
    
    # Initialise data.
    pollutionValues = np.zeros(shape=len(resolutions))
    dofs = np.zeros_like(pollutionValues)
    
    # Initialise physical parameters.
    U = utils.windTowardsReading
    params = Parameters(nu=10000, U=U)
    
    # Convergence loop.
    for i, res in enumerate(resolutions):
        
        # Create the mesh and record dof.
        mesh = UKMesh(res)
        dofs[i] = np.max(mesh.ID)+1 
        
        # Create the solver and run.
        solver = StaticSolver(mesh, params, utils.sotoFire)
        psi = solver.run()
        
        # Plot for sanity check.
        if debug == True:
            
            # filename = f"../plots/convergence_{res}.png"
            filename = f"plots/convergence_{res}.png"
            plt.plotGrid(mesh, psi, filename, saveFig=True)
        
        # Extract the solution at reading and store.
        pollutionValues[i] = utils.extractPollution(psi, mesh, utils.readingCoords)
    
    # We will get the error relative to the maximum resolution (1.25km).
    errors = abs(pollutionValues-pollutionValues[-1])[:-1]
    
    n = plotErrors(dofs, errors)
    print(n)

def plotErrors(dofs, errors):
    
    # Line of best fit.
    fit = np.polyfit(np.log(dofs[:-1]), np.log(errors), 1)
    
    # Plot the errors vs degrees of freedom (no of equations).    
    # filename = "../plots/convergenceError.png"
    filename = "plots/convergenceError.png"
    plt.plotErrors(dofs, errors, fit, filename, saveFig=True)
    
    return fit[0]

if __name__ == "__main__":
    
    runStaticConvergenceAnalysis()
    