
import os
import sys
sys.path.insert(0, './')

import logging
logging.getLogger('matplotlib').setLevel(logging.ERROR)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from tqdm import trange
from time import time
from mr_utils.sim.ssfp import ssfp
from mr_utils.recon.ssfp import gs_recon
from mr_utils import view

def mesh( height = 256, width = 512 ):

    # Material properties 
    PD = 1
    T1 = 100e-3
    T2 = 50e-3

    # Find indices 
    x = np.linspace(-1, 1, width)
    y = np.linspace(-1, 1, height)
    X, Y = np.meshgrid(x, y)

    # Fill in these indices with the values we want
    dims = X.shape
    PDs = np.zeros(dims)
    T1s = np.zeros(dims)
    T2s = np.zeros(dims)
    idx = X + Y > -2
    PDs[idx] = PD
    T1s[idx] = T1
    T2s[idx] = T2

    return { 'dims':dims, 'T2': T2s, 'T1' : T1s, 'PD' : PDs }

def simulation_phantom( height=256, 
                        width=512, 
                        nPC = 16, 
                        nC = 1,
                        TEs=[12e-3, 24e-3, 48e-3],
                        alpha = np.deg2rad(10)):
    
    # Calculate simulation values
    TRs = np.array(TEs) * 2.0;
    PCs = np.linspace(0, 2*np.pi, nPC, endpoint=False)
    
    # Get material properties and off-resonance values
    m = mesh(height, width);
    df_factor = 8;
    df_range = (-df_factor/np.max(TRs), df_factor/np.max(TRs))
    fx = np.linspace(df_range[0], df_range[1], width)
    fy = np.zeros(height)
    df, _ = np.meshgrid(fx, fy)

    # Simluate bSSFP for all coils and TRs and phase-cycles 
    M = np.zeros((nC, len(TRs), nPC, height, width), dtype='complex')
    for ii, TR in enumerate(TRs):
        for cc in range(nC):
            M[cc, ii, ...] = ssfp( m['T1'], m['T2'], TR, alpha, df, PCs, m['PD'])

    # Combine TR/phase-cycle dimension
    M = M.reshape((nC, len(TRs)*nPC, height, width))

    plt.imshow(abs(M[0,0,:]))
    plt.show()

    pass

if __name__ == '__main__':
    #mesh()
    simulation_phantom()