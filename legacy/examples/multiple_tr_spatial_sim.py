'''Example showing how to GASP using multiple PCs and TRs.

Notes
-----
Birdcage coil sensitivities are simulated to test effects of RF phase
offsets.  ncoils when >4 has significant performance impact as
animated plot GASPs in real time, leading to ncoil GASPs each
frame update.

TODO
----
- Find optimal values for TR and do performance analysis for npcs and
  nTRs.
- Try different forcing functions
'''
import sys
sys.path.insert(0, './')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from ssfp import bssfp as ssfp
from tqdm import trange

from gasp import gasp, get_cylinder, triangle_periodic as g
from gasp import generate_birdcage_sensitivities

if __name__ == '__main__':

    # Simulation parameters
    nTRs = 3 # number of TR
    TR_lo, TR_hi = 3e-3, 12e-3 # bounds of linspace for TRs
    npcs = 16 # number of phase-cycles at each TR
    ncoils = 2 # number of coils
    N = 128 # matrix size NxN
    C_dim = (2, N) # Calibration box - (# Number of lines of calibration, Pixels on signal)
    period = N / 2 # Period for forcing function
    bw = period / 2 # BW of forcing function

    # Experiment parameters
    TRs = np.linspace(TR_lo, TR_hi, nTRs) # Optimize these!
    alpha = np.deg2rad(35)
    pcs = np.linspace(0, 2*np.pi, npcs, endpoint=False)
    #pcs = np.linspace(-2*np.pi, 2*np.pi, npcs, endpoint=False)
    #pcs = np.linspace(0, 4*np.pi, npcs, endpoint=False)

    # Simple linear gradient off-resonance
    maxTR = np.max(TRs)
    minTR = np.min(TRs)
    df_range = (-1/maxTR, 1/maxTR)

    # Get a numerical phantom
    PD = 0.000040 # Adjust max magnitute to match phantom
    T1 = 100e-3
    T2 = 50e-3
    PDs, T1s, T2s, df = get_cylinder(
        N, df_range=df_range, radius=0.99, PD=PD, T1=T1, T2=T2)

    # Generate complex coil sensitivities
    csm = generate_birdcage_sensitivities(N, number_of_coils=ncoils)

    # Acquire all pcs at all TRs for all coils
    I = np.zeros((ncoils, nTRs, npcs, N, N), dtype='complex')
    for ii, TR in enumerate(TRs):
        for cc in trange(ncoils, leave=False):
            I[cc, ii, ...] = csm[cc, ...]*ssfp(
                T1s, T2s, TR, alpha, df, pcs, PDs)

            # # Correct phase profile
            '''
            Imag = np.abs(I[cc, ii, ...])
            Iphase = np.angle(I[cc, ii, ...]) - np.tile(
                pcs/2, (N, N, 1)).T
            I[cc, ii, ...] = Imag*np.exp(1j*Iphase)'''

    I0 = I[0, 0, :, int(N/2), int(N/2-60)]
    plt.subplot(2, 1, 1)
    plt.plot(np.abs(I0))
    plt.subplot(2, 1, 2)
    plt.plot(np.angle(I0))
    plt.show()

    # Combine TR/phase-cycle dimension
    I = I.reshape((ncoils, nTRs*npcs, N, N))

    # Do a neat thing a sweep across left to right while GASPing
    fig = plt.figure()
    im = plt.imshow(np.abs(I[0, 0, ...]), vmin=0, vmax=1)
    plt.title('Results of GASP swept across spatial extent')
    def animate(frame):
        '''Run plot update.'''

        # Construct the shifted spatial forcing function
        _D = g(N, period, frame, bw)

        # GASP for each coil
        Ic = np.zeros((ncoils, N, N), dtype='complex')
        for kk in range(ncoils):
            Ic[kk, ...] = gasp(I[kk, ...], _D, C_dim, pc_dim=0)

        # Do SOS and call it good
        _I = np.sqrt(np.sum(np.abs(Ic)**2, axis=0))
        im.set_data(_I)

        return im, # pylint: disable=R1707

    anim = animation.FuncAnimation(
        fig, animate, frames=N, blit=True, interval=0)
    plt.show()

    # Just look at a single slice of the first coil
    D0 = g(N, period, 0, bw)  #np.roll(D, int(N/5))
    I0 = gasp(I[0, ...], D0, C_dim, pc_dim=0)
    plt.plot(np.abs(I0[int(N/2), :]), label='Simulated Profile')
    plt.plot(D0, '--', label='Desired Profile')
    plt.legend()
    plt.title('Center horizontal slice\'s spatial response profile')
    plt.show()
