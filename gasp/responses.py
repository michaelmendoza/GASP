
import numpy as np

def triangle(x0, bw):
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