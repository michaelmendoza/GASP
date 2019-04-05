import math
import numpy as np
from scipy.signal import triang

def triangle(x0, bw):
    '''Spatial forcing function.

    Parameters
    ----------
    x0 : array_like
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

def triangle_periodic(img_width, period, offset, bw):

    assert period >= bw

    window = triang(bw)
    window = np.concatenate((window, np.zeros(period - bw)))
    num_of_windows = math.ceil(img_width / period)
    response = np.tile(window, (num_of_windows))
    response = response[:img_width]
    response = np.roll(response, offset)
    return response

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    plt.plot(triangle_periodic(128, 32, 8, 16))
    #plt.plot(triangle_periodic(128, 32, 8, 32))
    plt.show()