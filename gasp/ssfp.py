""" ssfp simulation """

import numpy as np


def ssfp(T1, T2, TR, TE, alpha, dphi=(0,), field_map=0, M0=1, f0=0, phi=0, useSqueeze=True) -> np.ndarray:
    """ Multiple acquisition ssfp """
    dphi = np.atleast_2d(dphi)

    M = []
    for ii, pc in np.ndenumerate(dphi):
        M.append(_ssfp(T1, T2, TR, TE, alpha, pc, field_map, M0, f0, phi)[..., None])
    M = np.concatenate(M, axis=-1)
    
    # Squeeze out dim of length 1, otherwise shape is [width, height, dphi]
    M = np.squeeze(M)
    if not useSqueeze and len(dphi) == 1:
        M = M[..., None]
    return M


def _ssfp(T1, T2, TR, TE, alpha, dphi=0, field_map=0, M0=1, f0=0, phi=0):
    """ transverse signal for ssfp mri after excitation at TE

    Parameters
    ----------
    T1 : float or array_like
        longitudinal exponential decay time constant (in seconds).
    T2 : float or array_like
        transverse exponential decay time constant (in seconds).
    TR : float
        repetition time (in seconds).
    alpha : float or array_like
        flip angle (in rad).
    dphi : float, optional
        Linear phase-cycle increment (in rad).
    field_map : float or array_like, optional
        B0 field map (in Hz).
    M0 : float or array_like, optional
        proton density.
    f0 : float, optional
        off-resonance (in Hz). Includes factors like the chemical shift
        of species w.r.t. the water peak.
    phi : float, optional
        phase offset (in rad).
    """
    
    # Convention for Ernst-Anderson based implementation from Hoff
    field_map = -1 * field_map
    
    # Set T1, T2, alpha, and field_map inputs to arrays
    T1 = np.atleast_2d(T1)
    T2 = np.atleast_2d(T2)
    alpha = np.atleast_2d(alpha)
    f0 = np.atleast_2d(f0)
    field_map = np.atleast_2d(field_map)

    # Compute exponential decay and handle T1, T2 of zero
    E1 = np.zeros(T1.shape)
    E1[T1 > 0] = np.exp(-TR/T1[T1 > 0])
    E2 = np.zeros(T2.shape)
    E2[T2 > 0] = np.exp(-TR/T2[T2 > 0])

    # Precompute theta and derivatives of theta and alpha
    beta = 2 * np.pi * (f0 + field_map) * TR
    theta = beta - dphi; # theta => phase per repetition time
    cos_alpha = np.cos(alpha)
    sin_alpha = np.sin(alpha)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    # Calculate Mxy 
    Mbottom = (1 - E1 * cos_alpha) * (1 - E2 * cos_theta) - E2 * (E1 - cos_alpha) * (E2 - cos_theta)
    Mx = M0 * (1 - E1) * sin_alpha * (1 - E2 * cos_theta) / Mbottom
    My = M0 * (1 - E1) * E2 * sin_alpha * sin_theta / Mbottom
    Mc = Mx + 1j * My

    # Add additional phase and handle T2 of zero 
    T2 = np.array(T2)
    idx = np.where(T2 > 0)
    val = np.zeros(T2.shape)
    val[idx] = -TE/T2[idx]
    _phi = beta * (TE / TR) + phi
    Mc = Mc * np.exp(1j * _phi) * np.exp(val)

    return Mc


def add_noise(I: np.ndarray, mu: float=0, sigma: float=0.005, factor: float=1) -> np.ndarray:
    """add gaussian noise to given simulated bSSFP signals

    Parameters
    ----------
    I: array_like
       images size(M,N,C)
    mu: float
        mean of the normal distribution
    sd: float
        standard deviation of the normal distribution

        Returns
    -------
    Mxy : numpy.ndarray
        Transverse complex magnetization with added .
    """
    noise = factor * np.random.normal(mu, sigma, (2,) + np.shape(I))
    noise_matrix = noise[0] + 1j*noise[1]
    return I + noise_matrix
