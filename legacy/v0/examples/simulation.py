
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

from gasp import gasp, gasp_coefficents, apply_gasp, triangle, triangle_periodic

def mesh( height = 256, width = 512, matIdx = 0 ):

    # Material properties 
    PD = 1.0
    T1a = 790e-3
    T2a = 92e-3
    f0a = 0
    T1b = 270e-3
    T2b = 85e-3
    f0b = 0 # -428 Hz @ 3T

    # Find indices 
    x = np.linspace(-1, 1, width)
    y = np.linspace(-1, 1, height)
    X, Y = np.meshgrid(x, y)

    # Fill in these indices with the values we want
    dims = X.shape
    PDs = np.zeros(dims)
    T1s = np.zeros(dims)
    T2s = np.zeros(dims)
    F0 = np.zeros(dims)

    if matIdx == 0:
        idx = X > -1
        PDs[idx] = PD
        T1s[idx] = T1a
        T2s[idx] = T2a  
        F0[idx] = f0a
    elif matIdx == 1:
        idx = X > -1
        PDs[idx] = PD
        T1s[idx] = T1b
        T2s[idx] = T2b
        F0[idx] = f0b   
    elif matIdx == 2:
        idx = X < 0
        PDs[idx] = PD
        T1s[idx] = T1a
        T2s[idx] = T2a
        F0[idx] = f0a
        idx = X > 0
        PDs[idx] = PD
        T1s[idx] = T1b
        T2s[idx] = T2b
        F0[idx] = f0b  
    elif matIdx == 3:
        idx = abs(X) < 0.5
        PDs[idx] = PD
        T1s[idx] = T1a
        T2s[idx] = T2a
        F0[idx] = f0a

    return { 'dims':dims, 'T2': T2s, 'T1' : T1s, 'PD' : PDs, 'F0': F0 }

def ssfp_phantom( height=256, 
                  width=512, 
                  nPC = 16, 
                  nC = 1,
                  TEs=[3e-3, 6e-3, 12e-3], #[12e-3, 24e-3, 48e-3],
                  alpha = np.deg2rad(35),
                  fieldGradDir = 1,
                  matIdx = 0 ):
    
    # Calculate simulation values
    TRs = np.array(TEs) * 2.0;
    PCs = np.linspace(0, 2*np.pi, nPC, endpoint=False)
    
    # Get material properties
    m = mesh(height, width, matIdx);
    df_factor = 2;
    df_range = (-df_factor/np.max(TRs), df_factor/np.max(TRs))

    # Get off-resonance values
    if(fieldGradDir):
        fx = np.linspace(df_range[0], df_range[1], width)
        fy = np.zeros(height)
        dfx, dfy = np.meshgrid(fx, fy)
        df = dfx + m['F0']
    else:
        fx = np.zeros(width)
        fy = np.linspace(df_range[0], df_range[1], height)
        dfx, dfy = np.meshgrid(fx, fy)
        df = dfy + m['F0']

    # Simluate bSSFP for all coils and TRs and phase-cycles 
    M = np.zeros((nC, len(TRs), nPC, height, width), dtype='complex')
    for ii, TR in enumerate(TRs):
        for cc in range(nC):
            M[cc, ii, ...] = ssfp( m['T1'], m['T2'], TR, alpha, df, PCs, m['PD'])

    # Combine TR/phase-cycle dimension
    M = M.reshape((nC, len(TRs)*nPC, height, width))

    #plt.imshow(abs(M[0,0,:]), cmap='gray')
    #plt.show() 

    return M

def gasp_coeff_phantom( height=256, 
                        width=512, 
                        nPC = 16, 
                        nC = 1, 
                        matIdx = 0 ):
    
    M = ssfp_phantom( height, width, nPC, nC, matIdx = 0 )
    D = triangle_periodic(width, 76, 18, 38)
    C_dim = (2, width)
    I, An = gasp_coefficents(M[0, ...], D, C_dim, pc_dim = 0 )

    '''
    plt.plot(np.abs(I[int(height/2), :]), label='Simulated Profile')
    plt.plot(D, '--', label='Desired Profile')
    plt.legend()
    plt.title('Center horizontal slice\'s spatial response profile')
    plt.show()
    '''

    return An, D

def simulation_phantom(height=256, 
                        width=512, 
                        nPC = 8, 
                        nC = 1):
    
    An, D = gasp_coeff_phantom( height, width, nPC, nC, matIdx = 0 )
    M = ssfp_phantom(height, width, nPC, nC, matIdx = 1)
    I = apply_gasp(M, An) 

    I = I / I.max() # Normalize for plotting 
    plt.plot(np.abs(I[int(height/2), :]), label='Simulated Profile')
    plt.plot(D, '--', label='Desired Profile')
    plt.legend()
    plt.title('Center horizontal slice\'s spatial response profile')
    plt.show()

if __name__ == '__main__':
    #mesh()
    #gasp_coeff_phantom();
    simulation_phantom()

