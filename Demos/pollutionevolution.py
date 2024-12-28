# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 23:28:05 2024

@author: mn1215
"""

import numpy as np

from src.grids import UKMesh
from src.parameters import Parameters
from src.solver import TemporalSolver
import src.utils as utils

import Demos.plotters as plt

def runPollutionEvolution():
    """ 
    Runs the evolution of pollution over Reading for different grid resolutions.
    No real analysis, just observing and having a good time."""
    
    debug=False
    
    # Initialise physical parameters.
    U = utils.windTowardsReading
    params = Parameters(nu=10000, U=U)
    
    # Solver parameters.
    endtime = 15000
    dt = 0.1 # It's fine.
    
    # Resolutions to iterate over.
    resolutions = ["40", "20", "10"]
    
    # Initialise data.
    numEntries = 200
    pollutionValues = np.zeros(shape=(len(resolutions), numEntries))
    
    # Iterate through resolutions.
    for i, res in enumerate(resolutions):
        
        # Generate mesh.
        mesh = UKMesh(res)
        
        # Create and run the temporal solver.
        solver = TemporalSolver(mesh, params, utils.sotoFire, dt, endtime)
        psi = solver.run()
        
        # Reduce psi to reasonable amount.
        psiReduced = psi[:, ::int(psi.shape[1]/(numEntries-1))]
        
        # Find the value of pollution at Reading for each time.
        for j in range(numEntries):
            
            if debug==True:
                plt.plotGrid(mesh, psiReduced[:, i])
            
            pollutionValues[i, j] = utils.extractPollution(psiReduced[:, j], 
                                                          mesh, utils.readingCoords)
    
    # Plot the results.
    # figname="../plots/pollutionEvolution.png"
    figname="plots/pollutionEvolution.png"
    plt.plotPollutionTimeseriesMultipleRes(pollutionValues, endtime, resolutions,
                                           figname, saveFig=True)

if __name__ == "__main__":
        
    runPollutionEvolution()