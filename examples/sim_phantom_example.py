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
# from ismrmrdtools.simulation import generate_birdcage_sensitivities
# from mr_utils.sim.ssfp import ssfp
from ssfp import bssfp as ssfp
# from mr_utils import view
from tqdm import trange, tqdm

from gasp import gasp, get_cylinder, triangle_periodic as g

if __name__ == '__main__':

    # Simulation parameters
    npcs = 16 # number of phase-cycles at each TR
    ncoils = 1 # number of coils
    height, width = 256, 512
    C_dim = (2, width) # Calibration box - (# Number of lines of calibration, Pixels on signal)
    period = 76 # Period for forcing function
    bw = 38 # BW of forcing function
    offset = 18

    # Experiment parameters
    TRs = [6e-3, 12e-3, 24e-3] # Optimize these!
    nTRs = len(TRs) # number of TR
    alpha = np.deg2rad(35)
    pcs = np.linspace(0, 2*np.pi, npcs, endpoint=False)
    #pcs = np.linspace(-2*np.pi, 2*np.pi, npcs, endpoint=False)
    #pcs = np.linspace(0, 4*np.pi, npcs, endpoint=False)

    # Simple linear gradient off-resonance
    maxTR = np.max(TRs)/20
    minTR = np.min(TRs)/20
    df_range = (-1/maxTR, 1/maxTR)

    # Get the actual off-resonance map
    coil_fm_gre = np.load('data/20190401_GASP_PHANTOM/coil_fm_gre.npy')

    # Get a numerical phantom
    PD = 0.000040 # Adjust max magnitute to match phantom
    T1 = 100e-3
    T2 = 50e-3
    PDs, T1s, T2s, _df = get_cylinder(width, df_range=df_range, radius=0.38, PD=PD, T1=T1, T2=T2)
    trim = int(width/4)
    PDs = PDs[trim:-trim, :]
    T1s = T1s[trim:-trim, :]
    T2s = T2s[trim:-trim, :]
    _df = _df[trim:-trim, :]
    _df = np.fliplr(_df)
    mask = T1s > 0

    print(_df.shape, coil_fm_gre[0, ...].shape)
    # from skimage.restoration import unwrap_phase
    TE1, TE2 = 2.87e-3, 5.74e-3
    fac = np.abs(TE1 - TE2)*2*np.pi
    # view(np.stack((_df*mask, coil_fm_gre[0, ...].T*mask)))
    # view(np.stack((_df*mask, unwrap_phase(fac*coil_fm_gre[0, ...].T*mask)/fac)))

    # Generate complex coil sensitivities -- let's skip this for now
    # csm = generate_birdcage_sensitivities(width, number_of_coils=ncoils)
    # csm = csm[:, trim:-trim, :]
    csm = np.ones(ncoils)

    # Acquire all pcs at all TRs for all coils
    I = np.zeros((ncoils, nTRs, npcs, height, width), dtype='complex')
    I_comp = I.copy()
    for ii, TR in tqdm(enumerate(TRs), leave=False, total=len(TRs)):
        for cc in trange(ncoils, leave=False):
            I_comp[cc, ii, ...] = csm[cc, ...]*ssfp(T1s, T2s, TR, alpha, coil_fm_gre[cc, ...].T*mask, pcs, PDs)
            # df0 = unwrap_phase(fac*coil_fm_gre[0, ...].T*mask)/fac
            # I[cc, ii, ...] = csm[cc, ...]*ssfp( T1s, T2s, TR, alpha, df0, pcs, PDs)
            I[cc, ii, ...] = csm[cc, ...]*ssfp( T1s, T2s, TR, alpha, _df, pcs, PDs)
            # print(np.stack((I[cc, ii, ...], I_comp[cc, ii, ...])).shape)
            # view(np.stack((I[cc, ii, ...], I_comp[cc, ii, ...], I[cc, ii, ...] - I_comp[cc, ii, ...])), fft_axes=(3, 4), montage_axis=0, movie_axis=1)

            # Compensate for phase accrual during the TE
            # I[cc, ii, ...] *= np.tile(np.exp(-2j * np.pi * coil_fm_gre[cc, ...].T * TR / 2), (npcs, 1, 1))
            # I[cc, ii, ...] *= np.tile(np.exp(-2j * np.pi * _df * TR / 2), (npcs, 1, 1))

    # Combine TR/phase-cycle dimension
    I = I.reshape((ncoils, nTRs*npcs, height, width))

    # Do a neat thing a sweep across left to right while GASPing
    fig = plt.figure()
    im = plt.imshow(np.abs(I[0, 0, ...]), vmin=0, vmax=1)
    plt.title('Results of GASP swept across spatial extent')
    def animate(frame):
        '''Run plot update.'''

        # Construct the shifted spatial forcing function
        _D = g(width, period, frame, bw)
        _D *= mask[int(height/2), :]

        # GASP for each coil
        Ic = np.zeros((ncoils, height, width), dtype='complex')
        for kk in range(ncoils):
            Ic[kk, ...] = gasp(I[kk, ...], _D, C_dim, pc_dim=0)

        # Do SOS and call it good
        _I = np.sqrt(np.sum(np.abs(Ic)**2, axis=0))
        im.set_data(_I)

        return im, # pylint: disable=R1707

    anim = animation.FuncAnimation(
        fig, animate, frames=width, blit=True, interval=0)
    plt.show()

    # Just look at a single slice of the first coil
    D0 = g(width, period, offset, bw)
    D0 *= mask[int(height/2), :]
    I0 = gasp(I[0, ...], D0, C_dim, pc_dim=0)
    plt.plot(np.abs(I0[int(height/2), :]), label='Simulated Profile')
    plt.plot(D0, '--', label='Desired Profile')
    plt.legend()
    plt.title('Center horizontal slice\'s spatial response profile')
    plt.show()
