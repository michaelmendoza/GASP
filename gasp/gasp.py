"""GASP module."""

import numpy as np


def gasp(I, D, C_dim, pc_dim: int=0):
    """Generation of Arbitrary Spectral Profiles.

    Parameters
    ----------
    I : array_like
        Array of phase-cycled images.
    D : array_like
        Vector of samples of desired spectral profile.
    C_dim: tuple
        Calibration box dimensions in number of pixels.
    pc_dim : int, optional
        Axis containing phase-cycles.

    Returns
    -------
    I0 : array_like
        Combined image with spatial response approximating D.
    """

    out, An = gasp_coefficients(I, D, C_dim, pc_dim)

    return out


def gasp_coefficients(I, D, C_dim, pc_dim: int=0):
    """Generation of Arbitrary Spectral Profiles.

    Parameters
    ----------
    I : array_like
        Array of phase-cycled images.
    D : array_like
        Vector of samples of desired spectral profile.
    C_dim: tuple
        Calibration box dimensions in number of pixels.
    pc_dim : int, optional
        Axis containing phase-cycles.

    Returns
    -------
    I0 : array_like
        Combined image with spatial response approximating D.
    """

    # Let's put the phase-cycle dimension last
    I = np.moveaxis(I, pc_dim, -1)
    I0 = I.copy()

    # Save the in-plane dimensions for reshape at end
    xx, yy = I.shape[:2]

    mid = [int(xx/2), int(yy/2)]
    pad = [int(C_dim[0]/2), int(C_dim[1]/2)]
    I = I[mid[0]-pad[0]:mid[0]+pad[0], mid[1]-pad[1]:mid[1]+pad[1], :]
    D = D[mid[1]-pad[1]:mid[1]+pad[1]]

    #view(I, movie_axis=-1)

    # Now let's put all the voxels' time curves down the first dim
    I = I.reshape((-1, I.shape[-1]))

    # Now repeat the desired spectral profile the correct number of
    # times to line up with the length of each column
    D = np.tile(D, (int(I.shape[0]/D.size),))
    # print(I.shape, D.shape)

    # Now solve the system
    x = np.linalg.lstsq(I, D, rcond=None)[0]
    # print(x.shape)

    out = I0.dot(x).reshape(xx, yy)

    return out, x

def apply_gasp(I, An):
    ''' Use gasp model on input magenatization data, I. Shape should be [height, width, PC x TRs]'''
    #I = np.squeeze(I).transpose(1, 2, 0)
    xx, yy = I.shape[0], I.shape[1]
    return I.dot(An).reshape(xx, yy)
