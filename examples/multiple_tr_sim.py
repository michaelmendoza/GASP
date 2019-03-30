'''Example showing how to GASP using multiple PCs and TRs.'''

import numpy as np
import matplotlib.pyplot as plt
from mr_utils.sim.ssfp import ssfp

from get_cylinder import get_cylinder #pylint: disable=C0413

if __name__ == '__main__':

    # Simulation parameters
    nTRs = 4
    npcs = 16
    N = 128

    # Experiment parameters
    TR0, TR1 = 3e-3, 12e-3
    TRs = np.linspace(TR0, TR1, nTRs) # Optimize these!
    print(TRs)
    alpha = np.deg2rad(30)
    pcs = np.linspace(-2*np.pi, 2*np.pi, npcs, endpoint=False)

    # Figure out minimum off-resonance we can resolve given the TRs
    # we chose
    maxTR = np.max(TRs)
    df_range = (-1/maxTR, 1/maxTR)

    # Get a numerical phantom
    PD, T1s, T2s, df = get_cylinder(N, df_range)

    # Here starts the funny business when you start using multiple
    # TRs...

    # What's the frequency resolution that we need to represent all
    # the TRs faithfully (i.e., all points we acquire fall on the
    # frequency grid during interpolation)?
    facs = (1/TRs)/(1/np.max(TRs))
    found_it = False
    for ii in range(1, 100):
        if np.all(np.fmod(facs*ii, 1) == 0):
            found_it = True
            break
    if not found_it:
        raise ValueError(('Could not find frequency resolution for '
                          'selected TRs!'))

    facs_ext = (facs*ii).astype(int)
    print(facs)
    print(facs_ext)

    # Do the sim over all TRs
    m = int(N/2)
    minTR = np.min(TRs)
    minTRext = minTR/np.min(facs_ext)
    maxfac = np.max(facs)
    df1 = np.linspace(-1/minTRext, 1/minTRext, npcs*np.max(facs_ext),
                      endpoint=False)

    # Synthesize all TRs, pcs into a common spectral profile
    I = np.zeros((nTRs, npcs*np.max(facs_ext), N, N), dtype='complex')
    for ii, TR in enumerate(TRs):
        I0 = ssfp(T1s, T2s, TR, alpha, df, pcs, PD)

        # periodically extend the signal to fit in (-1/minTR, 1/minTR)
        for x, y in np.ndindex((N, N)):
            Iext = np.tile(I0[:, m, m], (facs_ext[::-1][ii],))

            df0 = np.linspace(
                -1/minTRext, 1/minTRext, Iext.size, endpoint=False)

            # We want everyone to have the same frequenct axis, so
            # we'll have to do some interpolation to get there
            Iint_mag = np.interp(df1, df0, np.abs(Iext))
            Iint_pha = np.interp(df1, df0, np.angle(Iext))
            Iint = Iint_mag*np.exp(1j*Iint_pha)

            # Tail added by interpolation needs to be removed:
            # idx = facs_ext[::-1][ii]/2
            # Iint[-idx:] = np.nan

            # Stuff the interpolated, periodically extented signal
            # into the buffer
            I0[ii, :, x, y] = Iint

            # # Debug plot to make sure interpolation worked
            # plt.plot(df1, np.abs(Iint))
            # plt.plot(df0, np.abs(Iext), '*')
            # plt.show()

    # Now we need to construct our coefficient matrix with columns
    # made up of the flattened spectral profiles of each voxel for
    # a single image
    # A = I0.reshape()
