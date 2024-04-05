'''Forcing functions to use with GASP.'''

import math

import numpy as np
from scipy import signal
from scipy.stats import norm

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
    '''Spatial forcing function - Periodic Triangle with bandwidth

    Parameters
    ----------
    img_width : array_like
        Width of image singal
    period: int
        Period in pixels of forcing function
    offset: int
        Offset in pixels (i.e. cyclic shift)
    bw : float
        Bandwidth (width in pixel from base of triangle) of forcing
        function

    Returns
    -------
    g(x) : complex
        Desired spatial response of uniform phantom with periodic off
        resonance
    '''

    bw = round(bw)
    period = round(period)
    assert period >= bw

    window = signal.windows.triang(bw) # signal.triang(bw) <-- scipy v.0.12.0 
    window = np.concatenate((window, np.zeros(period - bw)))
    num_of_windows = math.ceil(img_width / period)
    response = np.tile(window, (num_of_windows))
    response = np.roll(response, offset)
    response = response[:img_width]
    return response

def square(width: int, bw: float, shift: float):
    ''' Square response curve
    width: number of pixels of response
    bw: width of bandpass as fraction of width: 0 to 1
    shift: shift of bandpass as fraction of width: -0.5 to 0.5
    '''
    if bw < 0:
        bw = 0
    if bw > 1:
        bw = 1

    if shift < -0.5:
        shift = -0.5
    if shift > 0.5:
        shift = 0.5

    bandpass = np.zeros(width)
    bw = width * bw
    x0 = shift * width
    xlo = int(width/2-bw/2+x0)
    xhi = int(width/2+bw/2+x0)
    bandpass[xlo:xhi] = 1
    return bandpass

def sinc(width, bw, shift):
    ''' Sinc response curve
    width: number of pixels of response
    bw: width of bandpass as fraction of width: 0 to 1
    shift: shift of bandpass as fraction of width: -0.5 to 0.5
    '''
    x0 = -1/bw * shift
    x = np.linspace(-1/bw + x0, 1/bw + x0, width)
    filter = np.sinc(x)
    return filter

def gaussian(width, bw, shift):
    ''' Gaussian response curve 
    width: number of pixels of response
    bw: width of gaussian: 0 to 1
    shift: shift of bandpass as fraction of width: -0.5 to 0.5
    '''
    sigma = bw / .2
    mu = 0
    x = np.linspace(-10, 10, width)
    y = norm.pdf(x, mu, sigma)
    y = np.roll(y, int(shift * width))
    y = y / max(y)
    return y

def notch(width, bw, shift):
    return 1 - gaussian(width, bw, shift)

def make_periodic(x, period:int = 2):
    length = x.shape[0]
    x = np.tile(x,(period))
    #x = np.roll(x, int(length / 2))
    return x