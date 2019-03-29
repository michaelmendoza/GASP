'''Example of how GASP works using simulated data.'''

import sys

import numpy as np
import matplotlib.pyplot as plt

from mr_utils.sim.ssfp import ssfp

sys.path.insert(0, './')
from gasp import gasp #pylint: disable=C0413

if __name__ == '__main__':

    # Experiment parameters
    TR = 10e-3
    alpha = np.deg2rad(10)
    npcs = 16
    pcs = np.linspace(-2*np.pi, 2*np.pi, npcs, endpoint=False)
    # pcs = 0

    # Tissue parameters
    T1 = 1.2
    T2 = .035
    M0 = 1
    df = 0
    # df = np.linspace(-1/TR, 1/TR, npcs, endpoint=False)
    phi_rf = np.deg2rad(0)

    # Simulate acquisition of phase-cycles
    I = ssfp(T1, T2, TR, alpha, df, pcs, M0, phi_rf=phi_rf)

    # Take a gander
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(np.abs(I), label='Magnitude')
    ax2.plot(np.rad2deg(np.angle(I)), '--', label='Phase')
    ax1.legend()
    ax1.set_ylabel('Magnitude')
    ax2.legend()
    ax2.set_ylabel('Phase')
    plt.show()

    # Make a box
    box = np.zeros(npcs)
    box[int(npcs/4):-int(npcs/4)] = 1
    # plt.plot(box)
    # plt.show()


    # Make the profile look like the box
    I0 = gasp(I, box)

    plt.plot(np.abs(I0))
    plt.plot(box, '--')
    plt.show()
