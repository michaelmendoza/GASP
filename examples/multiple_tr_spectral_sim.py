'''Example showing spectral shaping using multiple PCs and TRs.

This example is a little bit different than the spatial shaping
because we won't assume that we have a uniform phantom and thus we
can't rely on image space to give us information about the frequency
response of each voxel.

Not working currently, not sure if it's worth it because there will
need to be some adjusting for phase effects introduced by RF Tx/Rx
coils.
'''

import sys

import numpy as np
# import matplotlib.pyplot as pl
from ssfp import bssfp as ssfp

sys.path.insert(0, './')
from gasp import get_cylinder #pylint: disable=C0413


# We need a forcing function that will be compatible with
# the periodic spectra at each voxel in I along the phase-cycle
# dimension.  To do this we will parameterize the forcing
# function g(f) by frequency:
def g(f, maxTR):
    '''Frequency template function.

    Parameters
    ----------
    f : float
        Frequency (in Hz).
    maxTR : float
        The highest value TR that is acquired (in sec).

    Returns
    -------
    g(f) : complex
        Desired response at frequency f.
    '''
    # Naive triangle function implementation
    out = np.zeros(f.shape)
    for jj, ff in np.ndenumerate(f):
        if ff < -1/maxTR:
            out[jj] = 0
        elif ff > 1/maxTR:
            out[jj] = 0
        else:
            out[jj] = 1 - np.abs(ff)
    out[np.abs(out) > 0] -= np.min(out)
    return out



if __name__ == '__main__':

    # Simulation parameters
    nTRs = 4
    npcs = 8
    N = 16

    # Experiment parameters
    TR0, TR1 = 3e-3, 12e-3
    TRs = np.linspace(TR0, TR1, nTRs)
    print(TRs)
    alpha = np.deg2rad(30)
    pcs = np.linspace(-2*np.pi, 2*np.pi, npcs, endpoint=False)

    # Figure out minimum off-resonance we can resolve given the TRs
    # we chose
    maxTR = np.max(TRs)
    minTR = np.min(TRs)
    df_range = (-1/maxTR, 1/maxTR)

    # Get a numerical phantom
    PD, T1s, T2s, df = get_cylinder(N, df_range)

    #################################################################
    # Here starts the funny business when you start using multiple
    # TRs...
    #################################################################

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

            # We want everyone to have the same frequency axis, so
            # we'll have to do some interpolation to get there
            Iint_mag = np.interp(df1, df0, np.abs(Iext))
            Iint_pha = np.interp(df1, df0, np.angle(Iext))
            Iint = Iint_mag*np.exp(1j*Iint_pha)

            # Tail added by interpolation needs to be removed:
            # idx = facs_ext[::-1][ii]/2
            # Iint[-idx:] = np.nan

            # Stuff the interpolated, periodically extented signal
            # into the buffer
            I[ii, :, x, y] = Iint

            # # Debug plot to make sure interpolation worked
            # plt.plot(df1, np.abs(Iint))
            # plt.plot(df0, np.abs(Iext), '*')
            # plt.show()

        # # Debug plot to check out representative voxel from each TR
        # plt.plot(df1, np.abs(I[ii, :, m, m]))
        # plt.show()

    # Now we need to construct our coefficient matrix with columns
    # made up of the flattened spectral profiles of each voxel for
    # a single image
    # I = I.reshape()

    # The we need to find a coefficient for each voxel

    # Construct the image voxel-wise and reshape
