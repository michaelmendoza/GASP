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
from ismrmrdtools.simulation import generate_birdcage_sensitivities
from mr_utils.sim.ssfp import ssfp

from gasp import gasp, get_cylinder

def g(x0, bw):
    '''Spatial forcing function.

    Parameters
    ----------
    x0 : float
        Location (in px).
    bw : float
        Bandwidth of forcing function in Hz, e.g., 1/TR.

    Returns
    -------
    g(x) : complex
        Desired spatial response of uniform phantom.
    '''
    # Naive triangle function implementation
    out = np.zeros(x0.shape)
    for jj, xx in np.ndenumerate(x0):
        if xx < -bw:
            out[jj] = 0
        elif xx > bw:
            out[jj] = 0
        else:
            out[jj] = 1 - np.abs(xx)
    out[np.abs(out) > 0] -= np.min(out)
    return out/np.max(np.abs(out))

if __name__ == '__main__':

    # Simulation parameters
    nTRs = 4 # number of TR
    TR_lo, TR_hi = 3e-3, 12e-3 # bounds of linspace for TRs
    npcs = 6 # number of phase-cycles at each TR
    ncoils = 2 # number of coils
    N = 128 # matrix size NxN

    # Experiment parameters
    TRs = np.linspace(TR_lo, TR_hi, nTRs) # Optimize these!
    alpha = np.deg2rad(30)
    pcs = np.linspace(-2*np.pi, 2*np.pi, npcs, endpoint=False)

    # Simple linear gradient off-resonance
    maxTR = np.max(TRs)
    minTR = np.min(TRs)
    df_range = (-1/maxTR, 1/maxTR)

    # Get a numerical phantom
    PD, T1s, T2s, df = get_cylinder(N, df_range)

    # Generate complex coil sensitivities
    csm = generate_birdcage_sensitivities(N, number_of_coils=ncoils)

    # Acquire all pcs at all TRs for all coils
    I = np.zeros((ncoils, nTRs, npcs, N, N), dtype='complex')
    for ii, TR in enumerate(TRs):
        for cc in range(ncoils):
            I[cc, ii, ...] = csm[cc, ...]*ssfp(
                T1s, T2s, TR, alpha, df, pcs, PD)

            # Correct phase profile
            Imag = np.abs(I[cc, ii, ...])
            Iphase = np.angle(I[cc, ii, ...]) - np.tile(
                pcs/2, (N, N, 1)).T
            I[cc, ii, ...] = Imag*np.exp(1j*Iphase)

    # Combine TR/phase-cycle dimension
    I = I.reshape((ncoils, nTRs*npcs, N, N))

    # Do a neat thing a sweep across left to right while GASPing
    fig = plt.figure()
    im = plt.imshow(np.abs(I[0, 0, ...]), vmin=0, vmax=1)
    plt.title('Results of GASP swept across spatial extent')
    D = g(np.linspace(-1/minTR, 1/minTR, N), bw=1/(maxTR))
    def animate(frame):
        '''Run plot update.'''

        # Construct the shifted spatial forcing function
        _D = np.roll(D, -int(N/2) + frame)

        # GASP for each coil
        Ic = np.zeros((ncoils, N, N), dtype='complex')
        for kk in range(ncoils):
            Ic[kk, ...] = gasp(I[kk, ...], _D, pc_dim=0)

        # Do SOS and call it good
        _I = np.sqrt(np.sum(np.abs(Ic)**2, axis=0))
        im.set_data(_I)

        return im, # pylint: disable=R1707

    anim = animation.FuncAnimation(
        fig, animate, frames=N, blit=True, interval=0)
    plt.show()

    # Just look at a single slice of the first coil
    D0 = np.roll(D, int(N/5))
    I0 = gasp(I[0, ...], D0, pc_dim=0)
    plt.plot(np.abs(I0[int(N/2), :]), label='Simulated Profile')
    plt.plot(D0, '--', label='Desired Profile')
    plt.legend()
    plt.title('Center horizontal slice\'s spatial response profile')
    plt.show()
