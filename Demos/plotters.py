# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 23:35:30 2024

@author: mn1215
"""

import numpy as np
import matplotlib.pyplot as plt

import src.utils as utils

def plotGrid(mesh, psi, filename="", saveFig=False):
    
    plt.figure(figsize=(6, 4.5))
    plt.tripcolor(mesh.nodes[:, 0]/1e3, mesh.nodes[:, 1]/1e3, psi, triangles=mesh.IEN)
    plt.plot(utils.readingCoords[0]/1e3, utils.readingCoords[1]/1e3, '+r') # Reading location.
    plt.plot(utils.sotoCoords[0]/1e3, utils.sotoCoords[1]/1e3, '+r') # Soto location.
    plt.ylabel("Lat [km]")
    plt.xlabel("lon [km]")
    cbar = plt.colorbar()
    cbar.set_label("Pollution [normalised]")
    plt.tight_layout()
    
    if saveFig:
        plt.savefig(filename)
    
    plt.show()

def plotVelocityFieldOnGrid(mesh, U, filename="", saveFig=False):
    
    xmin, xmax = mesh.nodes[:, 0].min(), mesh.nodes[:, 0].max()
    ymin, ymax = mesh.nodes[:, 1].min(), mesh.nodes[:, 1].max()
    
    # Create a grid
    x = np.linspace(xmin, xmax, 15)
    y = np.linspace(ymin, ymax, 15)
    X, Y = np.meshgrid(x, y)
    
    VX = np.zeros_like(X)
    VY = np.zeros_like(Y)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            vx, vy = U([X[i, j], Y[i, j]])
            VX[i, j] = vx
            VY[i, j] = vy
    
    # Plot the velocity field
    plt.figure(figsize=(6, 6))
    plt.triplot(mesh.nodes[:, 0], mesh.nodes[:, 1], triangles=mesh.IEN, color="gray", alpha=0.7)
    plt.quiver(X, Y, -VX, -VY, angles="xy", scale_units="inches", scale=25, color="blue")  # Adjusted scale
    plt.scatter(*utils.readingCoords, color="red", label="Reading")
    plt.scatter(*utils.sotoCoords, color="blue", label="Southampton")
    plt.title("Cyclone-ish Velocity Field")
    plt.xlabel("Lon [m]")
    plt.ylabel("Lat [m]")
    plt.legend()
    plt.axis("equal")
    plt.tight_layout()
    
    if saveFig:
        plt.savefig(filename)
    plt.show()

def plotErrors(dofs, errors, fit, filename="", saveFig=False):
    
    slope, intercept = fit
    
    # plt.figure(figsize=(6, 6))
    plt.plot(dofs[:-1], errors, 'ko')
    plt.yscale("log")
    plt.xscale("log")
    plt.grid(which="both")
    plt.xlabel("DOF")
    plt.ylabel("Error")
    plt.tight_layout
    
    # Add line of best fit.
    lobf = np.exp(intercept) * dofs[:-1]**slope
    plt.plot(dofs[:-1], lobf, 'r-', label=f"n={slope:.2f}")
    plt.legend()
    
    if saveFig:
        plt.savefig(filename)
    
    plt.show()
    
def plotPollutionTimeseries(pollutionValues):
    
    plt.figure()
    plt.plot(pollutionValues)
    plt.show()
    
def plotPollutionTimeseriesMultipleRes(pollutionValues, endtime, labels, 
                                       figname="", saveFig=False):
    
    # Create time array.
    t = np.linspace(0, endtime, pollutionValues.shape[1])
    
    plt.figure()
    for i in range(pollutionValues.shape[0]):
        plt.plot(t/60**2, pollutionValues[i, :], label=labels[i])
    plt.legend()
    plt.grid()
    plt.xlabel("Time [hrs]")
    plt.ylabel("Pollution [normalised]")
    plt.tight_layout()
    
    if saveFig:
        plt.savefig(figname)
    
    plt.show()