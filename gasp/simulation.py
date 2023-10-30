import numpy as np
import matplotlib.pyplot as plt 
from skimage.filters import threshold_li
from gasp import ssfp, tissue, gasp as GASP
import warnings
warnings.simplefilter('ignore')

def simulate_gasp(D, width = 256, height = 256, npcs = 16, TRs = [5e-3, 10e-3, 15e-3], alpha = np.deg2rad(60), gradient = 2 * np.pi):
    ''' Simulates gasp with tissue phantom '''

    # Create phantoms, tissues, parameters
    t = tissue.tissue_generator(type='circle')
    mask = t['mask']
    size = mask.shape
    t1 = t['t1']
    t2 = t['t2']
    BetaMax = gradient
    beta = np.linspace(-BetaMax, BetaMax, size[1])
    f = beta / TRs[0] / (2 * np.pi)
    f = np.tile(f, (size[0], 1))
    pcs = np.linspace(0, 2*np.pi, npcs, endpoint=False)

    # Create simulated phantom data
    nTRs = len(TRs)
    M = np.empty((height, width, npcs,  nTRs), dtype=np.complex128)
    for ii, TR in enumerate(TRs):
        TE = TR / 2.0
        M[..., ii] = ssfp.ssfp(t1, t2, TR, TE, alpha, pcs, field_map=f, M0 = mask)
    M = np.reshape(M, (height, width, 1, npcs,  nTRs))
    
    # Create mask of phantom
    _ = np.sqrt(np.sum(np.abs(M)**2, axis=2))
    _ = np.mean(_, axis=2)
    _ = np.mean(_, axis=2)
    _ = abs(_)
    thresh = threshold_li(_)
    mask = np.abs(_) > thresh

    # Apply mask to data
    mask0 = np.tile(
        mask, (M.shape[2:] + (1, 1,))).transpose((3, 4, 0, 1, 2))
    data = M * mask0
    
    # Reshape data to required shapes
    data = np.reshape(data, data.shape[:-2] + (-1,))    # [Height, Width, Coil, PCs x TRs]
    data = np.moveaxis(data, 2, 0)                      # [Coil, Height, Width, PCs x TRs]
    data = data.transpose((0, 3, 1, 2))                 # [Coil,  PCs x TRs, Width, Height]

    # Get new dimensions
    ncoils, npcs, height, width = data.shape[:]

    # Calibration box - (# Number of lines of calibration, Pixels on signal)
    C_dim = (32, width)
    
    # Run gasp
    Ic = np.zeros((ncoils, height, width), dtype='complex')
    for cc in range(ncoils):
        Ic[cc, ...] = GASP.gasp(data[cc, ...], D, C_dim, pc_dim=0)
    Ic = np.sqrt(np.sum(np.abs(Ic)**2, axis=0))
    
    return Ic, M

def view_gasp(Ic, M, D):
    ''' Plots the results of gasp for given gasp output, Ic, input magnetization, M, 
        and target spectrum, D '''
    
    height = M.shape[0]
    width = M.shape[1]

    # Plot data
    _ = np.sqrt(np.sum(np.abs(M)**2, axis=2))
    _ = abs(_[:,:,0,0])

    f = plt.figure(figsize=(20,3))
    ax = f.add_subplot(141)
    ax2 = f.add_subplot(142)
    ax3 = f.add_subplot(143)
    ax4 = f.add_subplot(144)

    ax.imshow(_, cmap='gray')
    ax2.plot(_[int(width/2), :])
    ax3.imshow(Ic, cmap='gray')
    ax4.plot(np.abs(Ic[int(height/2), :]), label='Simulated Profile')
    ax4.plot(D, '--', label='Desired Profile')

    plt.show()

def view_gasp_analsys(G, D):
    length = len(G)

    f = plt.figure(figsize=(20,3))
    for i in range(length):
        ax = f.add_subplot(2, 8, i+1)
        ax2 = f.add_subplot(2, 8, 8 + i+1)

        g = G[i]
        d = D[i]
        ax.imshow(g, cmap='gray')
        ax2.plot(np.abs(g[int(g.shape[0]/2), :]), label='Simulated Profile')
        ax2.plot(d, '--', label='Desired Profile')