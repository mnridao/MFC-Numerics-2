# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 22:50:08 2024

@author: mn1215
"""

from src.grids import UKMesh
from src.parameters import Parameters
from src.solver import StaticSolver
import src.utils as utils

import plotters as plt

if __name__ == "__main__":
    
    saveFig=True
    
    # Setup mesh.
    mesh = UKMesh(resolution="1_25")
    
    U = utils.swirlyWhirlyWind
    params = Parameters(nu=2500, U=U)
    
    # Create the static solver and run.
    solver = StaticSolver(mesh, params, utils.sotoFire)
    psi = solver.run()
    
    # Plot the results.
    filename = "../plots/cycloneResults.png"
    plt.plotGrid(mesh, psi, filename, saveFig)
       
    filename = "../plots/swirlyVelocityField.png"
    plt.plotVelocityFieldOnGrid(mesh, U, filename, saveFig)