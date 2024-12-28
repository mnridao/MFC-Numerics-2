# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 00:33:05 2024

@author: mn1215
"""

from demos.convergence import runStaticConvergenceAnalysis
from demos.pollutionEvolution import runPollutionEvolution

if __name__ == "__main__":
    
    # This takes me less than 10 mins to run.
    runStaticConvergenceAnalysis()
    runPollutionEvolution()