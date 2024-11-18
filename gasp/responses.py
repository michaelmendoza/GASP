"""
responses.py: Standardized Filter Response Functions for Signal Processing

This module provides a collection of filter response functions commonly used in
signal processing and spectral analysis. Each function generates a response curve
with a standardized bandwidth and shift parameter. The available response types
include Square, Gaussian, Sinc, Lorentzian, and Butterworth filters.

The module also includes generic bandpass and stopband filter functions that can
use any of the implemented response types. All functions operate on a normalized
frequency range of [-1, 1] for consistency and easy comparison.

Functions:
    square, gaussian, sinc, lorentzian, butterworth: Individual filter responses
    bandpass: Generates a bandpass filter of the specified type
    stopband: Generates a stopband filter of the specified type

Usage:
    Import this module to access standardized filter response functions for
    various signal processing applications, spectral shaping, or filter design.
    For example useage, can run this module as a script to generate and plot all 
    response curves.
"""

import numpy as np
from scipy import signal

def square(width: int, bw: float, shift: float) -> np.ndarray:
    """
    Generate a square response curve with standardized bandwidth.

    Args:
        width (int): Number of points in the output array.
        bw (float): Bandwidth of the response, normalized to [0, 1].
        shift (float): Amount to shift the response, normalized to [-1, 1].

    Returns:
        np.ndarray: Square response curve.
    """
    bw = 2 * bw  # scale bandwidth to normalized frequency range
    x = np.linspace(-1, 1, width)
    y = np.zeros(width)
    y[np.abs(x) < bw/2] = 1
    return np.roll(y, int(shift * width))

def gaussian(width: int, bw: float, shift: float) -> np.ndarray:
    """
    Generate a Gaussian response curve with standardized bandwidth.

    Args:
        width (int): Number of points in the output array.
        bw (float): Bandwidth of the response, normalized to [0, 1].
        shift (float): Amount to shift the response, normalized to [-1, 1].

    Returns:
        np.ndarray: Gaussian response curve.
    """
    bw = 2 * bw  # scale bandwidth to normalized frequency range
    x = np.linspace(-1, 1, width)
    sigma = bw / 2.355  # FWHM = 2.355 * sigma for Gaussian
    y = np.exp(-(x**2) / (2 * sigma**2))
    return np.roll(y, int(shift * width))

def sinc(width: int, bw: float, shift: float) -> np.ndarray:
    """
    Generate a sinc response curve with standardized bandwidth.

    Args:
        width (int): Number of points in the output array.
        bw (float): Bandwidth of the response, normalized to [0, 1].
        shift (float): Amount to shift the response, normalized to [-1, 1].

    Returns:
        np.ndarray: Sinc response curve.
    """
    bw = 2 * bw  # scale bandwidth to normalized frequency range
    x = np.linspace(-1, 1, width)
    # The FWHM of an unscaled sinc is approximately 1.20784
    scale_factor = 1.20784 / bw
    y = np.sinc(x * scale_factor)
    return np.roll(y, int(shift * width))

def lorentzian(width: int, bw: float, shift: float) -> np.ndarray:
    """
    Generate a Lorentzian response curve with standardized bandwidth.

    Args:
        width (int): Number of points in the output array.
        bw (float): Bandwidth of the response, normalized to [0, 1].
        shift (float): Amount to shift the response, normalized to [-1, 1].

    Returns:
        np.ndarray: Lorentzian response curve.
    """
    bw = 2 * bw  # scale bandwidth to normalized frequency range
    x = np.linspace(-1, 1, width)
    gamma = bw / 2  # FWHM = 2 * gamma for Lorentzian
    y = 1 / (1 + (x / gamma)**2)
    return np.roll(y, int(shift * width))

def butterworth(width: int, bw: float, shift: float) -> np.ndarray:
    """
    Generate a Butterworth bandpass filter response with standardized bandwidth.

    Args:
        width (int): Number of points in the output array.
        bw (float): Bandwidth of the response, normalized to [0, 1].
        shift (float): Amount to shift the response, normalized to [-1, 1].

    Returns:
        np.ndarray: Butterworth bandpass filter response.
    """
    low = (1 - bw) / 2
    high = (1 + bw) / 2
    b, a = signal.butter(5, [low, high], btype='band')
    w, h = signal.freqz(b, a, worN=width)
    y = np.abs(h)
    return np.roll(y, int(shift * width))

def bandpass(width: int, bw: float, shift: float, type: str = 'butterworth') -> np.ndarray:
    """
    Generate a bandpass filter response with standardized bandwidth.

    Args:
        width (int): Number of points in the output array.
        bw (float): Bandwidth of the response, normalized to [0, 1].
        shift (float): Amount to shift the response, normalized to [-1, 1].
        type (str, optional): Type of bandpass filter. Defaults to 'butterworth'.
            Options: 'Square', 'Gaussian', 'Sinc', 'Lorentzian', 'Butterworth'.

    Returns:
        np.ndarray: Bandpass filter response of the specified type.

    Raises:
        ValueError: If an invalid filter type is specified.
    """
    responses = {
        'square': square,
        'gaussian': gaussian,
        'sinc': sinc,
        'lorentzian': lorentzian,
        'butterworth': butterworth
    }

    type = type.lower()
    if type not in responses:
        raise ValueError(f"Invalid filter type. Choose from: {', '.join(responses.keys())}")
    return responses[type](width, bw, shift)

def stopband(width: int, bw: float, shift: float, type: str = 'butterworth') -> np.ndarray:
    """
    Generate a stopband filter response with standardized bandwidth.

    Args:
        width (int): Number of points in the output array.
        bw (float): Bandwidth of the response, normalized to [0, 1].
        shift (float): Amount to shift the response, normalized to [-1, 1].
        type (str, optional): Type of stopband filter. Defaults to 'butterworth'.
            Options: 'Square', 'Gaussian', 'Sinc', 'Lorentzian', 'Butterworth'.

    Returns:
        np.ndarray: Stopband filter response of the specified type.
    """
    return 1 - bandpass(width, bw, shift, type)

if __name__ == "__main__":
    # Example usage
    
    def plot_all_responses(width: int, bw: float, shift: float):
        """ Generate and plot all response curves """

        x = np.linspace(-1, 1, width)
        responses = {
            'square': square(width, bw, shift),
            'gaussian': gaussian(width, bw, shift),
            'sinc': sinc(width, bw, shift),
            'lorentzian': lorentzian(width, bw, shift),
            'butterworth': butterworth(width, bw, shift)
        }

        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 8))
        for name, y in responses.items():
            plt.plot(x, y, label=name)

        plt.legend()
        plt.title(f'Response Functions (BW={bw}, Shift={shift})')
        plt.xlabel('Normalized Frequency')
        plt.ylabel('Amplitude')
        plt.grid(True)

        _3db = 10 ** (-3/20)
        plt.axhline(_3db, color='k', linestyle=':', label='-3 dB Line')
        plt.axhline(0.5, color='k', linestyle=':', label='-3 dB Line')
        plt.show()

    plot_all_responses(1000, 0.2, 0)