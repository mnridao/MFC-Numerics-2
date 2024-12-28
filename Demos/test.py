# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 15:55:26 2024

@author: mn1215
"""

import matplotlib.pyplot as plt
import numpy as np

from src.grids import UKMesh
from src.parameters import Parameters
from src.solver import StaticSolver, TemporalSolver
import src.utils as utils

def S(x):
    """
    Source term function representing pollution from the fire in Southampton. 
    Assumed Gaussian bump profile.

    Parameters:
    x (array-like): Global coordinates where the source term is to be evaluated.

    Returns:
    float: The value of the source term at the given coordinates.
    """
    sigma = 5000
    return np.exp(-1/(2*sigma**2)*((x[0]-442365)**2 + (x[1]-115483)**2))

# Reading coordinates.
coords = np.array([473993, 171625])
sotoCoords = np.array([442365, 115483])

if __name__ == "__main__":
    
    # Test the mesh.
    mesh = UKMesh('20')
    
    # Plot the grid.
    plt.triplot(mesh.nodes[:,0], mesh.nodes[:,1], triangles=mesh.IEN)
    # plt.plot(nodes[southern_boarder, 0], nodes[southern_boarder, 1], 'ro')
    plt.axis('equal')
    plt.show() 
    
    #%% Test the static solver.
    # U = lambda x: -10*np.array([1,0]) # North.
    U = lambda x: -10*np.array([1,2]) # Angle.
    params = Parameters(nu=10000, U=U)
    
    solver = StaticSolver(mesh, params, S)
    psi = solver.run()
    
    # Plot the result.
    plt.tripcolor(mesh.nodes[:, 0], mesh.nodes[:, 1], psi, triangles=mesh.IEN)
    plt.colorbar()
    plt.tight_layout()
    plt.show()

        
    pollution = utils.extractPollution(psi, mesh, coords)
    
    #%% Convergence test.
    
    # Resolutions to iterate over.
    resolutions = ["40", "20", "10", "5", "2_5", "1_25"]
    # resolutions = ["5", "2_5", "1_25"]
    
    # Initialise data.
    pollutionValues = np.zeros(shape=len(resolutions))
    dofs = np.zeros_like(pollutionValues)
    
    # Initialise physical parameters.
    # U = lambda x: -10*np.array([1,2]) # Angle.
    U = lambda x: -10*np.array([0.49082847, 0.87125623])
    
    def U(x):
        k = 1e5  # Scaling factor for velocity
        decay = 1e-5  # Decay factor for distance
        
        # Debug.
        k/=10
        decay*=2
        
        xc, yc = coords
        r = np.sqrt((x[0] - xc)**2 + (x[1] - yc)**2)
        r = np.maximum(r, 1e-10)  # Prevent division by zero
        vx = -k * (x[1] - yc) * decay / r
        vy = k * (x[0] - xc) * decay / r
        return -10*np.array([vx, vy])
    
    params = Parameters(nu=2500, U=U)
    params = Parameters(nu=10000, U=U)
    
    # Convergence loop.
    for i, res in enumerate(resolutions):
        
        # Create the mesh and record dof.
        mesh = UKMesh(res)
        dofs[i] = np.max(mesh.ID)+1 
        
        # Create the solver and run.
        solver = StaticSolver(mesh, params, S)
        psi = solver.run()
        
        # Plot for sanity check.
        plt.tripcolor(mesh.nodes[:, 0], mesh.nodes[:, 1], psi, triangles=mesh.IEN)
        plt.plot(coords[0], coords[1], '+r') # Reading location.
        plt.plot(sotoCoords[0], sotoCoords[1], '+r') # Soto location.
        plt.plot()
        plt.colorbar()
        plt.tight_layout()
        plt.show()
        
        # Extract the solution at reading and store.
        pollutionValues[i] = utils.extractPollution(psi, mesh, coords)
    
    # We will get the error relative to the maximum resolution (1.25km).
    errors = abs(pollutionValues-pollutionValues[-1])[:-1]
    
    #%%
    
    # Line of best fit.
    n, _ = np.polyfit(np.log(dofs[:-1]), np.log(errors), 1)
    
    # Plot the errors vs degrees of freedom (no of equations).    
    plt.figure()
    plt.plot(dofs[:-1], errors, 'ko')
    plt.yscale("log")
    plt.xscale("log")#
    plt.grid(which="both")
    plt.show()
    
    #%% Test the temporal solver.
    
    def U(x):
        k = 1e5  # Scaling factor for velocity
        decay = 1e-5  # Decay factor for distance
        
        # Debug.
        k/=10
        decay*=2
        
        xc, yc = coords
        r = np.sqrt((x[0] - xc)**2 + (x[1] - yc)**2)
        r = np.maximum(r, 1e-10)  # Prevent division by zero
        vx = -k * (x[1] - yc) * decay / r
        vy = k * (x[0] - xc) * decay / r
        return -20*np.array([vx, vy])
    
    U = lambda x: -10*np.array([0.49082847, 0.87125623])
    params = Parameters(nu=10000, U=U)
    
    mesh = UKMesh("20")
    
    endtime = 15000
    dt = 0.1
    
    solver = TemporalSolver(mesh, params, S, dt, endtime)
    phi = solver.run()
    
    #%%
    
    # Initialise data.
    pollutionValues = np.zeros(phi.shape[1])
    
    for i in range(1, phi.shape[1]):
        
        # Extract the pollution at reading.
        pollutionValues[i] = utils.extractPollution(phi[:, i], mesh, coords)
        
        # if i % 50000 == 0:
        #     plt.tripcolor(mesh.nodes[:, 0], mesh.nodes[:, 1], phi[:, i], triangles=mesh.IEN)
        #     plt.colorbar()
        #     plt.tight_layout()
        #     plt.show()
            
    #%%
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Define the center of the spiral
    coords = np.array([473993, 171625])
    
    # Parameters

    
    # Define the velocity field
    def velocity_field(x, y, center):
        xc, yc = center
        r = np.sqrt((x - xc)**2 + (y - yc)**2)
        r = np.maximum(r, 1e-10)  # Prevent division by zero
        vx = -k * (y - yc) * decay / r
        vy = k * (x - xc) * decay / r
        return vx, vy
    
    # Create a grid
    x = np.linspace(coords[0] - 10000, coords[0] + 10000, 20)
    y = np.linspace(coords[1] - 10000, coords[1] + 10000, 20)
    X, Y = np.meshgrid(x, y)
    
    # Compute the velocity field
    VX, VY = velocity_field(X, Y, coords)
    
    # Plot the velocity field
    plt.figure(figsize=(8, 8))
    plt.quiver(X, Y, VX, VY, angles="xy", scale_units="inches", scale=2, color="blue")
    plt.triplot(mesh.nodes[:,0], mesh.nodes[:,1], triangles=mesh.IEN)
    plt.scatter(*coords, color="red", label="Spiral Center")
    plt.title("Spiraling Velocity Field")
    plt.xlabel("x-coordinate")
    plt.ylabel("y-coordinate")
    plt.legend()
    plt.axis("equal")
    plt.show()
