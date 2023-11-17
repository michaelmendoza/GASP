import os
import numpy as np
import matplotlib.pyplot as plt 
from gasp import dataset, responses, simulation

if __name__ == "__main__":

    def gasp_example():
        width = 256
        height = 256
        alpha = np.deg2rad(30)
        gradient = 2 * np.pi
        phantom_type = 'circle'

        M = simulation.simulate_ssfp(alpha = alpha, gradient = gradient, phantom_type=phantom_type)
        D = responses.gaussian(width, bw=0.2, shift=0)
        Ic, M, An = simulation.simulate_gasp(D, alpha = alpha, gradient = gradient, phantom_type=phantom_type)

        simulation.view_gasp_results(Ic, M, D)

        gasp_example()
